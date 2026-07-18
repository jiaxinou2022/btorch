// connectome_bench — in-tree runner: load a Matrix Market case, run a registered
// implementation through the load -> preprocess -> compute lifecycle under the shared
// measurement layer, and write records in the unified schema.
//
//   connectome_bench --matrix path.mtx --target spmv --alg reference.eigen_csr \
//                    --batch 32 --warmup 10 --rep 50 --out results/cpp
#include <unistd.h>  // gethostname

#include <ctime>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "connectome_bench/loaders.h"
#include "connectome_bench/measure.h"
#include "connectome_bench/records.h"
#include "connectome_bench/registry.h"
#include "connectome_bench/runner.h"
#include "connectome_bench/writers.h"
#include "reference/reference.h"
#ifdef CONNECTOME_BENCH_HAVE_CUSPARSE
#include "cusparse/cusparse_baseline.h"
#endif
#ifdef CONNECTOME_BENCH_HAVE_SPUTNIK
#include "sputnik/sputnik_baseline.h"
#endif

namespace cb = connectome_bench;

namespace {

struct Args {
  std::string matrix;
  std::string target = "spmv";
  std::string alg = "reference.eigen_csr";
  std::string out = "results/cpp";
  std::string device = "cpu";
  std::string dtype = "float32";
  int batch = 1;
  int warmup = 10;
  int rep = 50;
};

Args ParseArgs(int argc, char** argv) {
  Args a;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string key = argv[i];
    const std::string val = argv[i + 1];
    if (key == "--matrix") a.matrix = val;
    else if (key == "--target") a.target = val;
    else if (key == "--alg") a.alg = val;
    else if (key == "--out") a.out = val;
    else if (key == "--device") a.device = val;
    else if (key == "--dtype") a.dtype = val;
    else if (key == "--batch") a.batch = std::stoi(val);
    else if (key == "--warmup") a.warmup = std::stoi(val);
    else if (key == "--rep") a.rep = std::stoi(val);
  }
  return a;
}

std::string NowIso() {
  std::time_t t = std::time(nullptr);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&t));
  return buf;
}

std::string Hostname() {
  char buf[256] = {0};
  return gethostname(buf, sizeof(buf) - 1) == 0 ? std::string(buf) : std::string();
}

// Impl names are "<provider>.<variant>" (e.g. "reference.eigen_csr").
std::string VariantOf(const std::string& impl_name) {
  const auto dot = impl_name.find('.');
  return dot == std::string::npos ? impl_name : impl_name.substr(dot + 1);
}

void ApplyTiming(cb::BenchmarkRecord& rec, const cb::MeasureResult& m) {
  rec.median_ms = m.median_ms;
  rec.p25_ms = m.p25_ms;
  rec.p75_ms = m.p75_ms;
  rec.min_ms = m.min_ms;
  rec.max_ms = m.max_ms;
  rec.compile_ms = m.compile_ms;
  rec.gbps = m.gbps;
  rec.gflops_s = m.gflops_s;
}

}  // namespace

int main(int argc, char** argv) {
  const Args args = ParseArgs(argc, argv);
  if (args.matrix.empty()) {
    std::cerr << "usage: connectome_bench --matrix PATH [--target spmv|spgemm] "
                 "[--alg NAME] [--batch N] [--warmup W] [--rep R] [--out DIR]\n";
    return 2;
  }

  cb::Registry registry;
  cb::RegisterBuiltins(registry);
#ifdef CONNECTOME_BENCH_HAVE_CUSPARSE
  cb::RegisterCusparse(registry);
#endif
#ifdef CONNECTOME_BENCH_HAVE_SPUTNIK
  cb::RegisterSputnik(registry);
#endif
  const cb::RunConfig cfg{args.device, args.dtype, args.warmup, args.rep};

  cb::BenchmarkRecord rec;
  rec.target = args.target;
  rec.alg = args.alg;
  rec.variant = VariantOf(args.alg);
  rec.device = args.device;
  rec.dtype = args.dtype;

  try {
    if (args.target == "spmv") {
      auto* impl = registry.FindSpmv(args.alg);
      if (!impl) throw std::runtime_error("unknown spmv impl: " + args.alg);
      rec.variant = VariantOf(impl->Name());
      const cb::SpmvCase c = cb::LoadSpmvCase(args.matrix);
      rec.case_id = c.case_id;
      rec.graph_id = c.graph_id;
      rec.n = c.n;
      rec.nnz = c.nnz;
      rec.density = c.density;
      rec.batch_size = args.batch;

      auto loaded = impl->LoadGraph(c, cfg);
      auto prepared = impl->Preprocess(*loaded, args.batch, cfg);
      cb::MeasureConfig mc;
      mc.warmup = args.warmup;
      mc.rep = args.rep;
      mc.total_flops = 2.0 * static_cast<double>(c.nnz) * args.batch;
      mc.io_bytes = static_cast<double>(c.nnz) * (sizeof(float) + sizeof(int)) +
                    (static_cast<double>(c.n) + 1) * sizeof(int) +
                    2.0 * static_cast<double>(c.n) * args.batch * sizeof(float);
      ApplyTiming(rec, cb::MeasureFunction([&] { impl->Compute(*prepared); }, mc));
    } else if (args.target == "spgemm") {
      auto* impl = registry.FindSpgemm(args.alg);
      if (!impl) throw std::runtime_error("unknown spgemm impl: " + args.alg);
      rec.variant = VariantOf(impl->Name());
      const cb::SpgemmCase c = cb::LoadSpgemmCase(args.matrix);
      rec.case_id = c.case_id;
      rec.graph_id = c.graph_id;
      rec.n = c.n;
      rec.nnz = c.nnz;
      rec.density = c.density;

      auto loaded = impl->LoadGraph(c, cfg);
      auto prepared = impl->Preprocess(*loaded, cfg);
      cb::MeasureConfig mc;
      mc.warmup = args.warmup;
      mc.rep = args.rep;
      mc.total_flops = 2.0 * static_cast<double>(c.intermediate_products);
      ApplyTiming(rec, cb::MeasureFunction([&] { impl->Compute(*prepared); }, mc));
    } else {
      throw std::runtime_error("unknown target: " + args.target);
    }
  } catch (const std::exception& e) {
    rec.status = "error";
    rec.error = e.what();
    std::cerr << "[error] " << e.what() << "\n";
  }

  const std::string run_id = args.target + "_" + args.alg;
  const auto run_dir = cb::EnsureRunDirectory(args.out, run_id);
  const std::vector<cb::BenchmarkRecord> records{rec};
  cb::WriteRecordsJsonl(records, run_dir / "records.jsonl");
  cb::WriteRecordsCsv(records, run_dir / "records.csv");

  cb::RunManifest manifest;
  manifest.run_id = run_id;
  manifest.target = args.target;
  manifest.entrypoint = "connectome_bench";
  manifest.created_at = NowIso();
  manifest.host = Hostname();  // produce-time host, for cross-machine provenance
  manifest.records_path = (run_dir / "records.jsonl").string();
  cb::WriteManifest(manifest, run_dir / "manifest.json");

  cb::EmitConsoleSummary(records);
  if (rec.median_ms) {
    std::cout << rec.target << " " << rec.alg << " on " << rec.graph_id
              << ": median " << *rec.median_ms << " ms";
    if (rec.gflops_s) std::cout << ", " << *rec.gflops_s << " GFLOP/s";
    std::cout << "\nrecords -> " << (run_dir / "records.jsonl").string() << "\n";
  }
  return rec.status == "ok" ? 0 : 1;
}
