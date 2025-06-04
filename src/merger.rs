use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

/// Allows writing structured records to mergeable input files
pub struct Writer {
    writer: BufWriter<File>,
    pub output_path: PathBuf,
}

impl Writer {
    pub fn new_with_suffix<P: AsRef<Path>>(prefix: P, suffix: u32) -> io::Result<Self> {
        let filename = format!("{}_{}.bin", prefix.as_ref().to_string_lossy(), suffix);
        let file = File::create(&filename)?;
        let writer = BufWriter::new(file);
        Ok(Self {
            writer,
            output_path: PathBuf::from(filename),
        })
    }

    pub fn write_record(&mut self, value: u32, count: u32, tags: &[u8], data: &[u8]) -> io::Result<()> {
        if tags.len() != count as usize * 4 || data.len() != count as usize * 64 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "mismatched tags/data/count"));
        }

        self.writer.write_all(&value.to_ne_bytes())?;
        self.writer.write_all(&count.to_ne_bytes())?;
        self.writer.write_all(tags)?;
        self.writer.write_all(data)?;
        Ok(())
    }
    
    pub fn finish(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

/// Represents one merged record
pub struct MergedEntry {
    pub value: u32,
    pub tags: Vec<u8>,
    pub data: Vec<u8>,
}

pub struct Merger {
    readers: Vec<BufReader<File>>,
    heap: BinaryHeap<HeapEntry>,
}

impl Merger {
    pub fn new(prefix: &str, count: usize) -> io::Result<Self> {
        let mut readers = Vec::with_capacity(count);
        let mut heap = BinaryHeap::new();

        for i in 0..count {
            let filename = format!("{}_{}.bin", prefix, i);
            let file = File::open(&filename)?;
            let mut reader = BufReader::new(file);
            if let Some(record) = read_record(&mut reader)? {
                heap.push(HeapEntry {
                    record,
                    source_index: i,
                });
                readers.push(reader);
            }
        }

        Ok(Merger { readers, heap })
    }
}

impl Iterator for Merger {
    type Item = io::Result<MergedEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.heap.pop()?;
        let mut combined_tags = current.record.tags;
        let mut combined_data = current.record.data;

        while let Some(next) = self.heap.peek() {
            if next.record.value != current.record.value {
                break;
            }

            let HeapEntry { record, source_index } = self.heap.pop().unwrap();
            combined_tags.extend(record.tags);
            combined_data.extend(record.data);

            match read_record(&mut self.readers[source_index]) {
                Ok(Some(following)) => self.heap.push(HeapEntry {
                    record: following,
                    source_index,
                }),
                Ok(None) => (),
                Err(e) => return Some(Err(e)),
            }
        }

        match read_record(&mut self.readers[current.source_index]) {
            Ok(Some(next_record)) => self.heap.push(HeapEntry {
                record: next_record,
                source_index: current.source_index,
            }),
            Ok(None) => (),
            Err(e) => return Some(Err(e)),
        }

        Some(Ok(MergedEntry {
            value: current.record.value,
            tags: combined_tags,
            data: combined_data,
        }))
    }
}

#[derive(Debug)]
struct Record {
    value: u32,
    tags: Vec<u8>,
    data: Vec<u8>,
}

#[derive(Debug)]
struct HeapEntry {
    record: Record,
    source_index: usize,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.record.value.cmp(&self.record.value)
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.record.value == other.record.value
    }
}
impl Eq for HeapEntry {}

fn read_record(reader: &mut BufReader<File>) -> io::Result<Option<Record>> {
    let value = match read_u32_native(reader) {
        Ok(v) => v,
        Err(ref e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e),
    };

    let count = read_u32_native(reader)?;

    let mut tags = vec![0u8; count as usize * 4];
    reader.read_exact(&mut tags)?;

    let mut data = vec![0u8; count as usize * 64];
    reader.read_exact(&mut data)?;

    Ok(Some(Record { value, tags, data }))
}

fn read_u32_native(reader: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(u32::from_ne_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_writer_and_merger_with_bytes() -> io::Result<()> {
        let prefix = "test_shard";
        let _ = fs::remove_file("test_shard_0.bin");
        let _ = fs::remove_file("test_shard_1.bin");

        let mut w0 = Writer::new_with_suffix(prefix, 0)?;
        let tags0 = 100u32.to_ne_bytes().into_iter().chain(101u32.to_ne_bytes()).collect::<Vec<_>>();
        let data0 = vec![0xAA; 128];
        w0.write_record(10, 2, &tags0, &data0)?;
        w0.finish()?;

        let mut w1 = Writer::new_with_suffix(prefix, 1)?;
        let tags1 = 200u32.to_ne_bytes().to_vec();
        let data1 = vec![0xBB; 64];
        w1.write_record(10, 1, &tags1, &data1)?;
        w1.finish()?;

        let merger = Merger::new(prefix, 2)?;
        let results: Vec<_> = merger.collect::<io::Result<_>>()?;

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].value, 10);
        assert_eq!(results[0].tags.len(), 12); // 3 * 4 bytes
        assert_eq!(results[0].data.len(), 192); // 3 * 64 bytes

        Ok(())
    }
}
