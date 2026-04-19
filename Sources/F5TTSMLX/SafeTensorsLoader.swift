import Foundation

// Minimal SafeTensors support for model-loading in this package.
// This parser focuses on metadata and raw byte extraction for tensor payloads.

enum SafeTensorDType: String, Sendable {
    case float16 = "F16"
    case bfloat16 = "BF16"
    case float32 = "F32"

    var byteWidth: Int {
        switch self {
        case .float16, .bfloat16:
            return 2
        case .float32:
            return 4
        }
    }
}

struct SafeTensorInfo: Sendable {
    let name: String
    let shape: [Int]
    let dtype: SafeTensorDType
    let dataRange: Range<Int>
}

enum SafeTensorsLoaderError: Error {
    case fileTooSmall
    case invalidHeaderLength
    case invalidHeaderJSON
    case malformedTensorEntry(String)
    case unsupportedDType(String)
    case invalidTensorOffsets(String)
    case tensorNotFound(String)
    case inconsistentTensorByteCount(String)
}

struct SafeTensorsLoader {
    private let fileData: Data
    private let dataSectionOffset: Int
    private let tensorsByName: [String: SafeTensorInfo]

    init(fileURL: URL) throws {
        let data = try Data(contentsOf: fileURL)
        try self.init(data: data)
    }

    // Data-based initializer makes unit testing simple and deterministic.
    init(data: Data) throws {
        guard data.count >= 8 else {
            throw SafeTensorsLoaderError.fileTooSmall
        }

        let headerLength = try Self.readUInt64LE(data: data, at: 0)
        guard headerLength <= UInt64(Int.max) else {
            throw SafeTensorsLoaderError.invalidHeaderLength
        }

        let headerLengthInt = Int(headerLength)
        let headerStart = 8
        let headerEnd = headerStart + headerLengthInt
        guard headerEnd <= data.count else {
            throw SafeTensorsLoaderError.invalidHeaderLength
        }

        let headerData = data.subdata(in: headerStart..<headerEnd)
        let jsonAny = try JSONSerialization.jsonObject(with: headerData)
        guard let root = jsonAny as? [String: Any] else {
            throw SafeTensorsLoaderError.invalidHeaderJSON
        }

        var entries: [String: SafeTensorInfo] = [:]
        entries.reserveCapacity(root.count)

        let dataOffset = headerEnd
        let dataLength = data.count - dataOffset

        for (name, value) in root {
            if name == "__metadata__" {
                continue
            }

            guard let object = value as? [String: Any] else {
                throw SafeTensorsLoaderError.malformedTensorEntry(name)
            }

            guard let dtypeRaw = object["dtype"] as? String,
                  let dtype = SafeTensorDType(rawValue: dtypeRaw)
            else {
                if let raw = (object["dtype"] as? String) {
                    throw SafeTensorsLoaderError.unsupportedDType(raw)
                }
                throw SafeTensorsLoaderError.malformedTensorEntry(name)
            }

            guard let shapeAny = object["shape"] as? [Any] else {
                throw SafeTensorsLoaderError.malformedTensorEntry(name)
            }

            let shape: [Int] = try shapeAny.map { element in
                if let intValue = element as? Int {
                    return intValue
                }
                if let number = element as? NSNumber {
                    return number.intValue
                }
                throw SafeTensorsLoaderError.malformedTensorEntry(name)
            }

            guard let offsets = object["data_offsets"] as? [Any], offsets.count == 2 else {
                throw SafeTensorsLoaderError.malformedTensorEntry(name)
            }

            guard let rawStart = Self.parseInt(offsets[0]),
                  let rawEnd = Self.parseInt(offsets[1]),
                  rawStart >= 0,
                  rawEnd >= rawStart,
                  rawEnd <= dataLength
            else {
                throw SafeTensorsLoaderError.invalidTensorOffsets(name)
            }

            let byteCount = rawEnd - rawStart
            let expectedByteCount = shape.reduce(1, *) * dtype.byteWidth
            if byteCount != expectedByteCount {
                throw SafeTensorsLoaderError.inconsistentTensorByteCount(name)
            }

            let dataRange = (dataOffset + rawStart)..<(dataOffset + rawEnd)
            entries[name] = SafeTensorInfo(name: name, shape: shape, dtype: dtype, dataRange: dataRange)
        }

        self.fileData = data
        self.dataSectionOffset = dataOffset
        self.tensorsByName = entries
    }

    var tensorNames: [String] {
        tensorsByName.keys.sorted()
    }

    func tensorInfo(named name: String) throws -> SafeTensorInfo {
        guard let info = tensorsByName[name] else {
            throw SafeTensorsLoaderError.tensorNotFound(name)
        }
        return info
    }

    func loadTensor(named name: String) throws -> (shape: [Int], dtype: SafeTensorDType, data: Data) {
        let info = try tensorInfo(named: name)
        let payload = fileData.subdata(in: info.dataRange)
        return (shape: info.shape, dtype: info.dtype, data: payload)
    }

    private static func readUInt64LE(data: Data, at offset: Int) throws -> UInt64 {
        guard offset >= 0, offset + 8 <= data.count else {
            throw SafeTensorsLoaderError.invalidHeaderLength
        }

        var value: UInt64 = 0
        _ = withUnsafeMutableBytes(of: &value) { destination in
            data.copyBytes(to: destination, from: offset..<(offset + 8))
        }
        return UInt64(littleEndian: value)
    }

    private static func parseInt(_ value: Any) -> Int? {
        if let intValue = value as? Int {
            return intValue
        }
        if let number = value as? NSNumber {
            return number.intValue
        }
        return nil
    }
}
