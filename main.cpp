#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <queue>
#include <condition_variable>
#include <memory>
#include <map>
#include <string>
#include <iomanip>

// Simple Huffman Coding Implementation
struct HuffmanNode {
    char character;
    int frequency;
    std::shared_ptr<HuffmanNode> left, right;
    
    HuffmanNode(char c, int freq) : character(c), frequency(freq), left(nullptr), right(nullptr) {}
    HuffmanNode(int freq) : character(0), frequency(freq), left(nullptr), right(nullptr) {}
};

struct Compare {
    bool operator()(const std::shared_ptr<HuffmanNode>& a, const std::shared_ptr<HuffmanNode>& b) {
        return a->frequency > b->frequency;
    }
};

class HuffmanEncoder {
private:
    std::map<char, std::string> codes;
    std::shared_ptr<HuffmanNode> root;
    
    void generateCodes(std::shared_ptr<HuffmanNode> node, const std::string& code) {
        if (!node) return;
        
        if (!node->left && !node->right) {
            codes[node->character] = code.empty() ? "0" : code;
            return;
        }
        
        generateCodes(node->left, code + "0");
        generateCodes(node->right, code + "1");
    }
    
    void serialize(std::shared_ptr<HuffmanNode> node, std::vector<char>& buffer) {
        if (!node) {
            buffer.push_back(0); // null marker
            return;
        }
        
        if (!node->left && !node->right) {
            buffer.push_back(1); // leaf marker
            buffer.push_back(node->character);
        } else {
            buffer.push_back(2); // internal node marker
            serialize(node->left, buffer);
            serialize(node->right, buffer);
        }
    }
    
    std::shared_ptr<HuffmanNode> deserialize(const std::vector<char>& buffer, size_t& pos) {
        if (pos >= buffer.size()) return nullptr;
        
        char marker = buffer[pos++];
        if (marker == 0) return nullptr;
        
        if (marker == 1) {
            char c = buffer[pos++];
            return std::make_shared<HuffmanNode>(c, 0);
        } else {
            auto node = std::make_shared<HuffmanNode>(0);
            node->left = deserialize(buffer, pos);
            node->right = deserialize(buffer, pos);
            return node;
        }
    }
    
public:
    void buildTree(const std::vector<char>& data) {
        std::map<char, int> frequency;
        for (char c : data) {
            frequency[c]++;
        }
        
        std::priority_queue<std::shared_ptr<HuffmanNode>, 
                          std::vector<std::shared_ptr<HuffmanNode>>, 
                          Compare> pq;
        
        for (auto& pair : frequency) {
            pq.push(std::make_shared<HuffmanNode>(pair.first, pair.second));
        }
        
        if (pq.size() == 1) {
            root = std::make_shared<HuffmanNode>(0);
            root->left = pq.top();
            pq.pop();
        } else {
            while (pq.size() > 1) {
                auto left = pq.top(); pq.pop();
                auto right = pq.top(); pq.pop();
                
                auto merged = std::make_shared<HuffmanNode>(left->frequency + right->frequency);
                merged->left = left;
                merged->right = right;
                
                pq.push(merged);
            }
            root = pq.top();
        }
        
        codes.clear();
        generateCodes(root, "");
    }
    
    std::vector<char> encode(const std::vector<char>& data) {
        std::vector<char> result;
        
        // Serialize tree structure
        std::vector<char> treeData;
        serialize(root, treeData);
        
        // Store tree size
        uint32_t treeSize = treeData.size();
        result.insert(result.end(), (char*)&treeSize, (char*)&treeSize + sizeof(uint32_t));
        result.insert(result.end(), treeData.begin(), treeData.end());
        
        // Encode data
        std::string bitString;
        for (char c : data) {
            bitString += codes[c];
        }
        
        // Convert bit string to bytes
        uint32_t originalSize = data.size();
        result.insert(result.end(), (char*)&originalSize, (char*)&originalSize + sizeof(uint32_t));
        
        for (size_t i = 0; i < bitString.length(); i += 8) {
            char byte = 0;
            for (int j = 0; j < 8 && i + j < bitString.length(); j++) {
                if (bitString[i + j] == '1') {
                    byte |= (1 << (7 - j));
                }
            }
            result.push_back(byte);
        }
        
        return result;
    }
    
    std::vector<char> decode(const std::vector<char>& encodedData) {
        size_t pos = 0;
        
        // Read tree size
        uint32_t treeSize = *(uint32_t*)&encodedData[pos];
        pos += sizeof(uint32_t);
        
        // Deserialize tree
        std::vector<char> treeData(encodedData.begin() + pos, encodedData.begin() + pos + treeSize);
        size_t treePos = 0;
        root = deserialize(treeData, treePos);
        pos += treeSize;
        
        // Read original size
        uint32_t originalSize = *(uint32_t*)&encodedData[pos];
        pos += sizeof(uint32_t);
        
        // Decode data
        std::vector<char> result;
        result.reserve(originalSize);
        
        auto current = root;
        for (size_t i = pos; i < encodedData.size() && result.size() < originalSize; i++) {
            char byte = encodedData[i];
            for (int bit = 7; bit >= 0 && result.size() < originalSize; bit--) {
                if (byte & (1 << bit)) {
                    current = current->right;
                } else {
                    current = current->left;
                }
                
                if (!current->left && !current->right) {
                    result.push_back(current->character);
                    current = root;
                }
            }
        }
        
        return result;
    }
};

// Thread-safe work queue
template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue;
    mutable std::mutex mutex;
    std::condition_variable condition;
    
public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex);
        queue.push(item);
        condition.notify_one();
    }
    
    bool try_pop(T& item) {
        std::lock_guard<std::mutex> lock(mutex);
        if (queue.empty()) return false;
        item = queue.front();
        queue.pop();
        return true;
    }
    
    void wait_and_pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex);
        while (queue.empty()) {
            condition.wait(lock);
        }
        item = queue.front();
        queue.pop();
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.empty();
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex);
        return queue.size();
    }
};

// Chunk structure for parallel processing
struct DataChunk {
    std::vector<char> data;
    size_t chunkIndex;
    size_t originalSize;
    
    DataChunk() : chunkIndex(0), originalSize(0) {}
    DataChunk(std::vector<char> d, size_t idx) : data(std::move(d)), chunkIndex(idx), originalSize(data.size()) {}
};

class MultithreadedCompressor {
private:
    static const size_t DEFAULT_CHUNK_SIZE = 64 * 1024; // 64KB chunks
    size_t numThreads;
    std::atomic<size_t> processedChunks{0};
    std::atomic<size_t> totalChunks{0};
    
    void compressWorker(ThreadSafeQueue<DataChunk>& inputQueue,
                       ThreadSafeQueue<DataChunk>& outputQueue,
                       std::atomic<bool>& finished) {
        HuffmanEncoder encoder;
        DataChunk chunk;
        
        while (!finished.load() || !inputQueue.empty()) {
            if (inputQueue.try_pop(chunk)) {
                encoder.buildTree(chunk.data);
                auto compressed = encoder.encode(chunk.data);
                
                DataChunk result(std::move(compressed), chunk.chunkIndex);
                result.originalSize = chunk.originalSize;
                outputQueue.push(std::move(result));
                
                processedChunks.fetch_add(1);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    
    void decompressWorker(ThreadSafeQueue<DataChunk>& inputQueue,
                         ThreadSafeQueue<DataChunk>& outputQueue,
                         std::atomic<bool>& finished) {
        HuffmanEncoder encoder;
        DataChunk chunk;
        
        while (!finished.load() || !inputQueue.empty()) {
            if (inputQueue.try_pop(chunk)) {
                auto decompressed = encoder.decode(chunk.data);
                
                DataChunk result(std::move(decompressed), chunk.chunkIndex);
                outputQueue.push(std::move(result));
                
                processedChunks.fetch_add(1);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    
    void progressReporter() {
        while (processedChunks.load() < totalChunks.load()) {
            size_t processed = processedChunks.load();
            size_t total = totalChunks.load();
            
            if (total > 0) {
                double progress = (double)processed / total * 100.0;
                std::cout << "\rProgress: " << std::fixed << std::setprecision(1) 
                         << progress << "% (" << processed << "/" << total << " chunks)";
                std::cout.flush();
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        std::cout << "\rProgress: 100.0% (" << totalChunks.load() << "/" 
                 << totalChunks.load() << " chunks) - Complete!" << std::endl;
    }
    
public:
    MultithreadedCompressor(size_t threads = 0) {
        if (threads == 0) {
            numThreads = std::thread::hardware_concurrency();
            if (numThreads == 0) numThreads = 4; // fallback
        } else {
            numThreads = threads;
        }
        
        std::cout << "Using " << numThreads << " threads for processing." << std::endl;
    }
    
    bool compressFile(const std::string& inputFile, const std::string& outputFile) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::ifstream inFile(inputFile, std::ios::binary);
        if (!inFile) {
            std::cerr << "Error: Cannot open input file: " << inputFile << std::endl;
            return false;
        }
        
        // Get file size
        inFile.seekg(0, std::ios::end);
        size_t fileSize = inFile.tellg();
        inFile.seekg(0, std::ios::beg);
        
        std::cout << "Compressing file: " << inputFile << " (" << fileSize << " bytes)" << std::endl;
        
        // Calculate chunks
        size_t chunkSize = std::max(DEFAULT_CHUNK_SIZE, fileSize / (numThreads * 4));
        totalChunks.store((fileSize + chunkSize - 1) / chunkSize);
        processedChunks.store(0);
        
        ThreadSafeQueue<DataChunk> inputQueue, outputQueue;
        std::atomic<bool> readFinished{false};
        
        // Read and queue chunks
        std::thread readerThread([&]() {
            size_t chunkIndex = 0;
            while (inFile) {
                std::vector<char> buffer(chunkSize);
                inFile.read(buffer.data(), chunkSize);
                size_t bytesRead = inFile.gcount();
                
                if (bytesRead > 0) {
                    buffer.resize(bytesRead);
                    inputQueue.push(DataChunk(std::move(buffer), chunkIndex++));
                }
            }
            readFinished.store(true);
        });
        
        // Start worker threads
        std::vector<std::thread> workers;
        for (size_t i = 0; i < numThreads; i++) {
            workers.emplace_back(&MultithreadedCompressor::compressWorker, this,
                               std::ref(inputQueue), std::ref(outputQueue), std::ref(readFinished));
        }
        
        // Progress reporter
        std::thread progressThread(&MultithreadedCompressor::progressReporter, this);
        
        // Collect results
        std::map<size_t, DataChunk> results;
        size_t collectedChunks = 0;
        
        while (collectedChunks < totalChunks.load()) {
            DataChunk result;
            outputQueue.wait_and_pop(result);
            results[result.chunkIndex] = std::move(result);
            collectedChunks++;
        }
        
        // Wait for all threads
        readerThread.join();
        for (auto& worker : workers) {
            worker.join();
        }
        progressThread.join();
        
        // Write compressed file
        std::ofstream outFile(outputFile, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error: Cannot create output file: " << outputFile << std::endl;
            return false;
        }
        
        // Write header
        uint64_t originalFileSize = fileSize;
        uint64_t numChunks = totalChunks.load();
        outFile.write((char*)&originalFileSize, sizeof(uint64_t));
        outFile.write((char*)&numChunks, sizeof(uint64_t));
        
        // Write chunk sizes
        for (size_t i = 0; i < numChunks; i++) {
            uint32_t chunkSize = results[i].data.size();
            uint32_t originalSize = results[i].originalSize;
            outFile.write((char*)&chunkSize, sizeof(uint32_t));
            outFile.write((char*)&originalSize, sizeof(uint32_t));
        }
        
        // Write compressed data
        size_t totalCompressedSize = 0;
        for (size_t i = 0; i < numChunks; i++) {
            outFile.write(results[i].data.data(), results[i].data.size());
            totalCompressedSize += results[i].data.size();
        }
        
        inFile.close();
        outFile.close();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        double compressionRatio = (double)totalCompressedSize / fileSize;
        double spaceSaved = (1.0 - compressionRatio) * 100.0;
        
        std::cout << "\nCompression completed successfully!" << std::endl;
        std::cout << "Original size: " << fileSize << " bytes" << std::endl;
        std::cout << "Compressed size: " << totalCompressedSize << " bytes" << std::endl;
        std::cout << "Compression ratio: " << std::fixed << std::setprecision(2) 
                 << compressionRatio << " (" << spaceSaved << "% space saved)" << std::endl;
        std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(2)
                 << (fileSize / 1024.0 / 1024.0) / (duration.count() / 1000.0) << " MB/s" << std::endl;
        
        return true;
    }
    
    bool decompressFile(const std::string& inputFile, const std::string& outputFile) {
        auto start = std::chrono::high_resolution_clock::now();
        
        std::ifstream inFile(inputFile, std::ios::binary);
        if (!inFile) {
            std::cerr << "Error: Cannot open compressed file: " << inputFile << std::endl;
            return false;
        }
        
        // Read header
        uint64_t originalFileSize, numChunks;
        inFile.read((char*)&originalFileSize, sizeof(uint64_t));
        inFile.read((char*)&numChunks, sizeof(uint64_t));
        
        std::cout << "Decompressing file: " << inputFile << std::endl;
        std::cout << "Expected output size: " << originalFileSize << " bytes" << std::endl;
        
        totalChunks.store(numChunks);
        processedChunks.store(0);
        
        // Read chunk metadata
        std::vector<std::pair<uint32_t, uint32_t>> chunkInfo(numChunks);
        for (size_t i = 0; i < numChunks; i++) {
            inFile.read((char*)&chunkInfo[i].first, sizeof(uint32_t));  // compressed size
            inFile.read((char*)&chunkInfo[i].second, sizeof(uint32_t)); // original size
        }
        
        ThreadSafeQueue<DataChunk> inputQueue, outputQueue;
        std::atomic<bool> readFinished{false};
        
        // Read compressed chunks
        std::thread readerThread([&]() {
            for (size_t i = 0; i < numChunks; i++) {
                std::vector<char> buffer(chunkInfo[i].first);
                inFile.read(buffer.data(), chunkInfo[i].first);
                inputQueue.push(DataChunk(std::move(buffer), i));
            }
            readFinished.store(true);
        });
        
        // Start worker threads
        std::vector<std::thread> workers;
        for (size_t i = 0; i < numThreads; i++) {
            workers.emplace_back(&MultithreadedCompressor::decompressWorker, this,
                               std::ref(inputQueue), std::ref(outputQueue), std::ref(readFinished));
        }
        
        // Progress reporter
        std::thread progressThread(&MultithreadedCompressor::progressReporter, this);
        
        // Collect results
        std::map<size_t, DataChunk> results;
        size_t collectedChunks = 0;
        
        while (collectedChunks < numChunks) {
            DataChunk result;
            outputQueue.wait_and_pop(result);
            results[result.chunkIndex] = std::move(result);
            collectedChunks++;
        }
        
        // Wait for all threads
        readerThread.join();
        for (auto& worker : workers) {
            worker.join();
        }
        progressThread.join();
        
        // Write decompressed file
        std::ofstream outFile(outputFile, std::ios::binary);
        if (!outFile) {
            std::cerr << "Error: Cannot create output file: " << outputFile << std::endl;
            return false;
        }
        
        for (size_t i = 0; i < numChunks; i++) {
            outFile.write(results[i].data.data(), results[i].data.size());
        }
        
        inFile.close();
        outFile.close();
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "\nDecompression completed successfully!" << std::endl;
        std::cout << "Output size: " << originalFileSize << " bytes" << std::endl;
        std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
        std::cout << "Throughput: " << std::fixed << std::setprecision(2)
                 << (originalFileSize / 1024.0 / 1024.0) / (duration.count() / 1000.0) << " MB/s" << std::endl;
        
        return true;
    }
    
    void benchmarkPerformance(const std::string& testFile) {
        std::cout << "\n=== Performance Benchmark ===" << std::endl;
        
        // Test with different thread counts
        std::vector<size_t> threadCounts = {1, 2, 4, numThreads};
        
        for (size_t threads : threadCounts) {
            if (threads > std::thread::hardware_concurrency()) continue;
            
            MultithreadedCompressor compressor(threads);
            
            std::cout << "\nTesting with " << threads << " thread(s):" << std::endl;
            
            auto start = std::chrono::high_resolution_clock::now();
            
            std::string compressedFile = testFile + ".compressed." + std::to_string(threads);
            if (compressor.compressFile(testFile, compressedFile)) {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                std::cout << "Compression time with " << threads << " threads: " 
                         << duration.count() << " ms" << std::endl;
                
                // Clean up
                std::remove(compressedFile.c_str());
            }
        }
    }
};

int main(int argc, char* argv[]) {
    std::cout << "=== Multithreaded File Compressor ===" << std::endl;
    std::cout << "Hardware concurrency: " << std::thread::hardware_concurrency() << " threads" << std::endl << std::endl;
    
    if (argc < 2) {
        std::cout << "Usage:" << std::endl;
        std::cout << "  " << argv[0] << " compress <input_file> [output_file] [threads]" << std::endl;
        std::cout << "  " << argv[0] << " decompress <input_file> [output_file] [threads]" << std::endl;
        std::cout << "  " << argv[0] << " benchmark <test_file>" << std::endl;
        std::cout << "\nExamples:" << std::endl;
        std::cout << "  " << argv[0] << " compress document.txt document.compressed" << std::endl;
        std::cout << "  " << argv[0] << " decompress document.compressed document_restored.txt" << std::endl;
        std::cout << "  " << argv[0] << " benchmark large_file.txt" << std::endl;
        return 1;
    }
    
    std::string operation = argv[1];
    
    if (operation == "benchmark" && argc >= 3) {
        std::string testFile = argv[2];
        MultithreadedCompressor compressor;
        compressor.benchmarkPerformance(testFile);
        return 0;
    }
    
    if (argc < 3) {
        std::cerr << "Error: Input file not specified." << std::endl;
        return 1;
    }
    
    std::string inputFile = argv[2];
    std::string outputFile = (argc >= 4) ? argv[3] : "";
    size_t threads = (argc >= 5) ? std::stoul(argv[4]) : 0;
    
    MultithreadedCompressor compressor(threads);
    
    if (operation == "compress") {
        if (outputFile.empty()) {
            outputFile = inputFile + ".compressed";
        }
        
        if (compressor.compressFile(inputFile, outputFile)) {
            std::cout << "\nCompression successful! Output: " << outputFile << std::endl;
        } else {
            std::cerr << "Compression failed!" << std::endl;
            return 1;
        }
    }
    else if (operation == "decompress") {
        if (outputFile.empty()) {
            outputFile = inputFile + ".decompressed";
        }
        
        if (compressor.decompressFile(inputFile, outputFile)) {
            std::cout << "\nDecompression successful! Output: " << outputFile << std::endl;
        } else {
            std::cerr << "Decompression failed!" << std::endl;
            return 1;
        }
    }
    else {
        std::cerr << "Error: Unknown operation '" << operation << "'" << std::endl;
        std::cerr << "Use 'compress', 'decompress', or 'benchmark'" << std::endl;
        return 1;
    }
    
    return 0;
}