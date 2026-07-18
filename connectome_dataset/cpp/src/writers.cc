#include "connectome_bench/writers.h"

#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>

namespace connectome_bench {

namespace {

std::string EscapeJson(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (char c : value) {
    switch (c) {
      case '\\': escaped += "\\\\"; break;
      case '"': escaped += "\\\""; break;
      case '\n': escaped += "\\n"; break;
      default: escaped += c; break;
    }
  }
  return escaped;
}

// JSON field helpers: emit `"key":value` with a trailing comma.
std::string JStr(const std::string& key, const std::string& value) {
  return "\"" + key + "\":\"" + EscapeJson(value) + "\",";
}

// Full round-trip precision for doubles; integers stay integers (no 6-sig-fig
// truncation of n/nnz on large connectome graphs).
constexpr int kDoubleDigits = std::numeric_limits<double>::max_digits10;

std::string JInt(const std::string& key, std::int64_t value) {
  return "\"" + key + "\":" + std::to_string(value) + ",";
}

std::string JNum(const std::string& key, double value) {
  std::ostringstream os;
  os.precision(kDoubleDigits);
  os << "\"" << key << "\":" << value << ",";
  return os.str();
}

template <typename T>
std::string JOpt(const std::string& key, const std::optional<T>& value) {
  std::ostringstream os;
  os.precision(kDoubleDigits);
  os << "\"" << key << "\":";
  if (value.has_value()) {
    os << *value;
  } else {
    os << "null";
  }
  os << ",";
  return os.str();
}

std::string JOptStr(const std::string& key, const std::optional<std::string>& value) {
  if (value.has_value()) return JStr(key, *value);
  return "\"" + key + "\":null,";
}

std::string RecordToJson(const BenchmarkRecord& r) {
  std::string s = "{";
  s += JStr("target", r.target);
  s += JStr("alg", r.alg);
  s += JStr("variant", r.variant);
  s += JStr("case_id", r.case_id);
  s += JStr("graph_id", r.graph_id);
  s += JInt("replicate", r.replicate);
  s += JStr("device", r.device);
  s += JStr("dtype", r.dtype);
  s += JInt("n", r.n);
  s += JInt("nnz", r.nnz);
  s += JNum("density", r.density);
  s += JOpt("batch_size", r.batch_size);
  s += JOpt("timesteps", r.timesteps);
  s += JOptStr("pass_name", r.pass_name);
  s += JOpt("median_ms", r.median_ms);
  s += JOpt("p25_ms", r.p25_ms);
  s += JOpt("p75_ms", r.p75_ms);
  s += JOpt("min_ms", r.min_ms);
  s += JOpt("max_ms", r.max_ms);
  s += JOpt("compile_ms", r.compile_ms);
  s += JOpt("gbps", r.gbps);
  s += JOpt("gflops_s", r.gflops_s);
  s += JOpt("cpu_delta_mb", r.cpu_delta_mb);
  s += JOpt("gpu_delta_mb", r.gpu_delta_mb);
  s += JStr("status", r.status);
  s += JStr("error", r.error);
  s += "\"extra_json\":" + (r.extra_json.empty() ? std::string("{}") : r.extra_json);
  s += "}";
  return s;
}

std::string CsvCell(const std::string& value) {
  std::string out = "\"";
  for (char c : value) {
    if (c == '"') out += "\"\"";  // RFC 4180: embedded quotes are doubled
    else out += c;
  }
  out += "\"";
  return out;
}

template <typename T>
std::string CsvOpt(const std::optional<T>& value) {
  if (!value.has_value()) return "";
  std::ostringstream os;
  os << *value;
  return os.str();
}

}  // namespace

void WriteRecordsJsonl(const std::vector<BenchmarkRecord>& records,
                       const std::filesystem::path& path) {
  std::ofstream out(path);
  for (const auto& record : records) {
    out << RecordToJson(record) << "\n";
  }
}

void WriteRecordsCsv(const std::vector<BenchmarkRecord>& records,
                     const std::filesystem::path& path) {
  std::ofstream out(path);
  out << "target,alg,variant,case_id,graph_id,replicate,device,dtype,n,nnz,density,"
         "batch_size,timesteps,pass_name,median_ms,p25_ms,p75_ms,min_ms,max_ms,"
         "compile_ms,gbps,gflops_s,status,error\n";
  for (const auto& r : records) {
    out << CsvCell(r.target) << ',' << CsvCell(r.alg) << ',' << CsvCell(r.variant) << ','
        << CsvCell(r.case_id) << ',' << CsvCell(r.graph_id) << ',' << r.replicate << ','
        << CsvCell(r.device) << ',' << CsvCell(r.dtype) << ',' << r.n << ',' << r.nnz << ','
        << r.density << ',' << CsvOpt(r.batch_size) << ',' << CsvOpt(r.timesteps) << ','
        << CsvCell(r.pass_name.value_or("")) << ',' << CsvOpt(r.median_ms) << ','
        << CsvOpt(r.p25_ms) << ',' << CsvOpt(r.p75_ms) << ',' << CsvOpt(r.min_ms) << ','
        << CsvOpt(r.max_ms) << ',' << CsvOpt(r.compile_ms) << ',' << CsvOpt(r.gbps) << ','
        << CsvOpt(r.gflops_s) << ',' << CsvCell(r.status) << ',' << CsvCell(r.error) << '\n';
  }
}

void WriteManifest(const RunManifest& manifest, const std::filesystem::path& path) {
  std::ofstream out(path);
  out << "{\n"
      << "  \"run_id\": \"" << EscapeJson(manifest.run_id) << "\",\n"
      << "  \"target\": \"" << EscapeJson(manifest.target) << "\",\n"
      << "  \"entrypoint\": \"" << EscapeJson(manifest.entrypoint) << "\",\n"
      << "  \"created_at\": \"" << EscapeJson(manifest.created_at) << "\",\n"
      << "  \"command\": \"" << EscapeJson(manifest.command) << "\",\n"
      << "  \"host\": \"" << EscapeJson(manifest.host) << "\",\n"
      << "  \"platform\": \"" << EscapeJson(manifest.platform) << "\",\n"
      << "  \"records_path\": \"" << EscapeJson(manifest.records_path) << "\"\n"
      << "}\n";
}

}  // namespace connectome_bench
