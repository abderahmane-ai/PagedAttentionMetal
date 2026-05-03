import Foundation
import Metal
import QuartzCore

// MARK: - Timing

struct Timer {
    private var start: CFAbsoluteTime = 0
    
    mutating func begin() {
        start = CFAbsoluteTime(CACurrentMediaTime())
    }
    
    func elapsed() -> Double {
        return CFAbsoluteTime(CACurrentMediaTime()) - start
    }
}

func measure<T>(_ iterations: Int = 1, _ block: () -> T) -> (result: T, avgMs: Double, minMs: Double, maxMs: Double) {
    var times: [Double] = []
    var result: T!
    
    for _ in 0..<iterations {
        var timer = Timer()
        timer.begin()
        result = block()
        times.append(timer.elapsed() * 1000)
    }
    
    let avg = times.reduce(0, +) / Double(times.count)
    return (result, avg, times.min() ?? 0, times.max() ?? 0)
}

// MARK: - Results

struct BenchmarkResult {
    let name: String
    let config: String
    let avgMs: Double
    let minMs: Double
    let maxMs: Double
    let throughput: Double?
    let memoryMB: Double?
    
    func formatted() -> String {
        var parts = [
            "[\(name)]",
            config,
            String(format: "avg: %.2fms", avgMs),
            String(format: "min: %.2fms", minMs),
            String(format: "max: %.2fms", maxMs)
        ]
        
        if let throughput = throughput {
            parts.append(String(format: "throughput: %.0f tok/s", throughput))
        }
        
        if let memoryMB = memoryMB {
            parts.append(String(format: "memory: %.1f MB", memoryMB))
        }
        
        return parts.joined(separator: " | ")
    }
    
    func csv() -> String {
        let throughputStr = throughput.map { String(format: "%.0f", $0) } ?? ""
        let memoryStr = memoryMB.map { String(format: "%.1f", $0) } ?? ""
        return "\(name),\(config),\(String(format: "%.2f", avgMs)),\(String(format: "%.2f", minMs)),\(String(format: "%.2f", maxMs)),\(throughputStr),\(memoryStr)"
    }
}

class ResultsTable {
    var results: [BenchmarkResult] = []
    
    func add(_ result: BenchmarkResult) {
        results.append(result)
    }
    
    func print() {
        Swift.print("\n" + String(repeating: "=", count: 100))
        for result in results {
            Swift.print(result.formatted())
        }
        Swift.print(String(repeating: "=", count: 100) + "\n")
    }
    
    func saveCSV(to path: String) throws {
        var csv = "Name,Config,Avg(ms),Min(ms),Max(ms),Throughput(tok/s),Memory(MB)\n"
        for result in results {
            csv += result.csv() + "\n"
        }
        try csv.write(toFile: path, atomically: true, encoding: .utf8)
    }
    
    func saveJSON(to path: String) throws {
        let dict = results.map { result -> [String: Any] in
            var d: [String: Any] = [
                "name": result.name,
                "config": result.config,
                "avg_ms": result.avgMs,
                "min_ms": result.minMs,
                "max_ms": result.maxMs
            ]
            if let throughput = result.throughput {
                d["throughput_tok_per_sec"] = throughput
            }
            if let memory = result.memoryMB {
                d["memory_mb"] = memory
            }
            return d
        }
        
        let data = try JSONSerialization.data(withJSONObject: dict, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: path))
    }
}

// MARK: - Progress

struct ProgressBar {
    let total: Int
    var current: Int = 0
    
    mutating func increment(_ label: String = "") {
        current += 1
        let percentage = Double(current) / Double(total) * 100
        let filled = Int(percentage / 2)
        let bar = String(repeating: "█", count: filled) + String(repeating: "░", count: 50 - filled)
        print("\r[\(bar)] \(Int(percentage))% \(label)", terminator: "")
        fflush(stdout)
        if current == total {
            print()
        }
    }
}
