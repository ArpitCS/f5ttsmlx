import Foundation

// Internal tokenizer skeleton mapped to the tokenizer and vocab handling in
// the Python f5-tts-mlx repository.
struct Tokenizer {
    private let vocabulary: [String: Int]
    private let sortedCompositeTokens: [String]

    init(vocabulary: [String: Int]) {
        self.vocabulary = vocabulary
        self.sortedCompositeTokens = vocabulary.keys
            .filter { $0.count > 1 }
            .sorted { lhs, rhs in
                if lhs.count != rhs.count {
                    return lhs.count > rhs.count
                }
                return lhs < rhs
            }
    }

    init(modelDirectory: URL) throws {
        let vocabURL = modelDirectory.appendingPathComponent("vocab.txt", isDirectory: false)
        let contents = try String(contentsOf: vocabURL, encoding: .utf8)

        // Python f5-tts-mlx reads one token per line and uses line index as id.
        // We approximate that behavior directly by preserving file order.
        let entries = contents.components(separatedBy: .newlines)
        var mapping: [String: Int] = [:]
        mapping.reserveCapacity(entries.count)

        for (index, token) in entries.enumerated() {
            if token.isEmpty {
                continue
            }
            mapping[token] = index
        }

        self.init(vocabulary: mapping)
    }

    /// Tokenizes text using a vocabulary loaded from `vocab.txt`.
    ///
    /// This is an approximate Swift counterpart to the Python repository's
    /// vocab-based indexing path: we greedily match known multi-character
    /// tokens first, then fall back to single-character lookup.
    func tokenize(_ text: String) -> [Int] {
        guard !text.isEmpty else {
            return []
        }

        let characters = Array(text)
        var result: [Int] = []
        result.reserveCapacity(characters.count)
        var index = 0

        while index < characters.count {
            var matchedLength = 0
            var matchedId: Int?

            for token in sortedCompositeTokens {
                let tokenChars = Array(token)
                let end = index + tokenChars.count
                if end > characters.count {
                    continue
                }
                if Array(characters[index..<end]) == tokenChars, let id = vocabulary[token] {
                    matchedLength = tokenChars.count
                    matchedId = id
                    break
                }
            }

            if let matchedId {
                result.append(matchedId)
                index += matchedLength
                continue
            }

            let single = String(characters[index])
            if let id = vocabulary[single] {
                result.append(id)
            }

            index += 1
        }

        return result
    }

    func forward(_ text: String) -> [Int] {
        tokenize(text)
    }
}
