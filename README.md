
# ⚡ Multithreaded File Compressor using Huffman Coding

A high-performance C++ application that compresses and decompresses files using Huffman encoding with multithreading for speed optimization. This tool demonstrates the impact of parallelism in file processing by chunking large files and processing them concurrently.

---

## 🚀 Features

- 🔁 **Lossless compression** using Huffman Coding
- 🧵 **Multithreaded processing** for parallel chunk compression/decompression
- 📈 **Built-in benchmarking** to compare single-threaded vs. multi-threaded performance
- 🧠 Smart chunking strategy for memory efficiency
- 🧼 Clean code using RAII, STL, and smart pointers
- 🧪 Real-time progress monitoring

---

## 🧩 How It Works

- The file is split into chunks (default 64KB or auto-adjusted).
- Each chunk is processed in parallel by worker threads.
- A Huffman tree is built per chunk to ensure lossless and independent decoding.
- Chunks are recombined and metadata is written into the compressed file for accurate decompression.

---

## 📦 Usage

```bash
./compressor [operation] <input_file> [output_file] [threads]
````

### 🔹 Operations

| Command      | Description                             |
| ------------ | --------------------------------------- |
| `compress`   | Compress a file                         |
| `decompress` | Decompress a previously compressed file |
| `benchmark`  | Benchmark compression performance       |

### 📌 Examples

```bash
# Compress with default threads
./compressor compress myfile.txt

# Decompress
./compressor decompress myfile.txt.compressed

# Compress with custom output and 8 threads
./compressor compress data.csv output.cmp 8

# Benchmark with various threads
./compressor benchmark large_file.txt
```

---

## 📊 Benchmark Sample

| Threads | Time (ms) | Compression Ratio | Throughput (MB/s) |
| ------- | --------- | ----------------- | ----------------- |
| 1       | 1100      | 0.45              | 8.1               |
| 2       | 600       | 0.45              | 15.0              |
| 4       | 350       | 0.45              | 25.7              |
| 8       | 280       | 0.45              | 32.1              |

> 📈 Results may vary depending on file type and hardware.

---

## 🛠️ Build Instructions

### Requirements

* C++17 or higher
* GCC / Clang / MSVC
* CMake (optional)

### Compile with g++

```bash
g++ -std=c++17 -O3 -pthread compressor.cpp -o compressor
```

Or use CMake:

```bash
mkdir build && cd build
cmake ..
make
```

---

## 📁 File Structure

```
.
├── compressor.cpp        # Main source code
├── README.md             # Project documentation
├── test_data/            # Sample input/output files
└── build/                # Optional build directory
```

---

## 📚 Concepts Demonstrated

* Huffman Tree construction & encoding
* Bit-level data packing/unpacking
* Thread synchronization (`mutex`, `condition_variable`)
* Smart pointers (`std::shared_ptr`, `std::unique_ptr`)
* Atomic counters for thread-safe progress tracking



## 👩‍💻 Author

**Prativa Dhar**
*Computer Science Student, Enthusiastic about Systems Programming and AI*


---

## 🙌 Acknowledgements

* [Huffman Coding Algorithm](https://en.wikipedia.org/wiki/Huffman_coding)
* STL Threads and Concurrency Utilities
* File Compression Projects from GitHub inspirations

---

