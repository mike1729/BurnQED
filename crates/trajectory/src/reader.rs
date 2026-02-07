//! Reads TrajectoryRecords from Parquet files.

use crate::types::{TrajectoryLabel, TrajectoryRecord, TrajectorySummary};
use arrow::array::*;
use arrow::record_batch::RecordBatch;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Static methods for reading trajectory data from Parquet files.
pub struct TrajectoryReader;

impl TrajectoryReader {
    /// Read all trajectory records from a Parquet file.
    pub fn read_all(path: &Path) -> anyhow::Result<Vec<TrajectoryRecord>> {
        let file = std::fs::File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        let mut records = Vec::new();
        for batch_result in reader {
            let batch = batch_result?;
            let mut batch_records = extract_records_from_batch(&batch)?;
            records.append(&mut batch_records);
        }

        tracing::debug!(
            count = records.len(),
            path = %path.display(),
            "Read trajectory records"
        );

        Ok(records)
    }

    /// Read trajectory records from multiple Parquet files.
    pub fn read_multiple(paths: &[PathBuf]) -> anyhow::Result<Vec<TrajectoryRecord>> {
        let mut all_records = Vec::new();
        for path in paths {
            let mut records = Self::read_all(path)?;
            all_records.append(&mut records);
        }
        Ok(all_records)
    }

    /// Compute summary statistics from a trajectory Parquet file.
    pub fn read_summary(path: &Path) -> anyhow::Result<TrajectorySummary> {
        let records = Self::read_all(path)?;

        let mut positive_count = 0;
        let mut negative_count = 0;
        let mut theorem_names = HashSet::new();
        let mut proved_theorems = HashSet::new();

        for record in &records {
            match record.label {
                TrajectoryLabel::Positive => positive_count += 1,
                TrajectoryLabel::Negative => negative_count += 1,
                TrajectoryLabel::Unknown => {}
            }
            theorem_names.insert(record.theorem_name.clone());
            if record.is_proof_complete {
                proved_theorems.insert(record.theorem_name.clone());
            }
        }

        Ok(TrajectorySummary {
            total_records: records.len(),
            positive_count,
            negative_count,
            unique_theorems: theorem_names.len(),
            proved_theorems: proved_theorems.len(),
        })
    }

    /// Read only records for a specific theorem from a Parquet file.
    pub fn read_for_theorem(path: &Path, theorem_name: &str) -> anyhow::Result<Vec<TrajectoryRecord>> {
        let records = Self::read_all(path)?;
        Ok(records
            .into_iter()
            .filter(|r| r.theorem_name == theorem_name)
            .collect())
    }
}

/// Extract trajectory records from a single Arrow RecordBatch.
fn extract_records_from_batch(batch: &RecordBatch) -> anyhow::Result<Vec<TrajectoryRecord>> {
    let theorem_names = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow::anyhow!("Column 0 (theorem_name) is not StringArray"))?;

    let state_ids = batch
        .column(1)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| anyhow::anyhow!("Column 1 (state_id) is not UInt64Array"))?;

    let state_pps = batch
        .column(2)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow::anyhow!("Column 2 (state_pp) is not StringArray"))?;

    let tactics = batch
        .column(3)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow::anyhow!("Column 3 (tactic_applied) is not StringArray"))?;

    let parent_ids = batch
        .column(4)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| anyhow::anyhow!("Column 4 (parent_state_id) is not UInt64Array"))?;

    let labels = batch
        .column(5)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow::anyhow!("Column 5 (label) is not StringArray"))?;

    let depths = batch
        .column(6)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| anyhow::anyhow!("Column 6 (depth_from_root) is not UInt32Array"))?;

    let remaining = batch
        .column(7)
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| anyhow::anyhow!("Column 7 (remaining_depth) is not Int32Array"))?;

    let log_probs = batch
        .column(8)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| anyhow::anyhow!("Column 8 (llm_log_prob) is not Float64Array"))?;

    let ebm_scores = batch
        .column(9)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| anyhow::anyhow!("Column 9 (ebm_score) is not Float64Array"))?;

    let proof_complete = batch
        .column(10)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| anyhow::anyhow!("Column 10 (is_proof_complete) is not BooleanArray"))?;

    let timestamps = batch
        .column(11)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| anyhow::anyhow!("Column 11 (timestamp_ms) is not UInt64Array"))?;

    let mut records = Vec::with_capacity(batch.num_rows());
    for i in 0..batch.num_rows() {
        let parent_state_id = if parent_ids.is_null(i) {
            None
        } else {
            Some(parent_ids.value(i))
        };

        records.push(TrajectoryRecord {
            theorem_name: theorem_names.value(i).to_string(),
            state_id: state_ids.value(i),
            state_pp: state_pps.value(i).to_string(),
            tactic_applied: tactics.value(i).to_string(),
            parent_state_id,
            label: TrajectoryLabel::from_str_lossy(labels.value(i)),
            depth_from_root: depths.value(i),
            remaining_depth: remaining.value(i),
            llm_log_prob: log_probs.value(i),
            ebm_score: ebm_scores.value(i),
            is_proof_complete: proof_complete.value(i),
            timestamp_ms: timestamps.value(i),
        });
    }

    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::TrajectoryLabel;
    use crate::writer::TrajectoryWriter;
    use tempfile::TempDir;

    fn make_test_record(state_id: u64, label: TrajectoryLabel) -> TrajectoryRecord {
        TrajectoryRecord {
            theorem_name: "test_theorem".to_string(),
            state_id,
            state_pp: format!("⊢ state_{state_id}"),
            tactic_applied: if state_id == 0 {
                String::new()
            } else {
                format!("tactic_{state_id}")
            },
            parent_state_id: if state_id == 0 {
                None
            } else {
                Some(state_id - 1)
            },
            label,
            depth_from_root: state_id as u32,
            remaining_depth: -1,
            llm_log_prob: -0.5 * state_id as f64,
            ebm_score: 0.0,
            is_proof_complete: false,
            timestamp_ms: 1700000000000 + state_id,
        }
    }

    #[test]
    fn test_roundtrip_write_read() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("roundtrip.parquet");

        let mut writer = TrajectoryWriter::new(path.clone());
        for i in 0..100 {
            let label = if i % 3 == 0 {
                TrajectoryLabel::Positive
            } else {
                TrajectoryLabel::Negative
            };
            writer.record(make_test_record(i, label));
        }
        writer.finish().unwrap();

        let records = TrajectoryReader::read_all(&path).unwrap();
        assert_eq!(records.len(), 100);

        // Verify first and last records
        assert_eq!(records[0].state_id, 0);
        assert_eq!(records[0].theorem_name, "test_theorem");
        assert_eq!(records[0].label, TrajectoryLabel::Positive);
        assert!(records[0].parent_state_id.is_none());

        assert_eq!(records[99].state_id, 99);
        assert_eq!(records[99].state_pp, "⊢ state_99");
        assert_eq!(records[99].tactic_applied, "tactic_99");
        assert_eq!(records[99].parent_state_id, Some(98));
    }

    #[test]
    fn test_nullable_parent_state_id() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("nullable.parquet");

        let mut writer = TrajectoryWriter::new(path.clone());

        // Root with None parent
        let mut r0 = make_test_record(0, TrajectoryLabel::Unknown);
        r0.parent_state_id = None;
        writer.record(r0);

        // Child with Some parent
        let mut r1 = make_test_record(1, TrajectoryLabel::Unknown);
        r1.parent_state_id = Some(0);
        writer.record(r1);

        // Another root with None parent
        let mut r2 = make_test_record(2, TrajectoryLabel::Unknown);
        r2.parent_state_id = None;
        writer.record(r2);

        writer.finish().unwrap();

        let records = TrajectoryReader::read_all(&path).unwrap();
        assert_eq!(records.len(), 3);
        assert!(records[0].parent_state_id.is_none());
        assert_eq!(records[1].parent_state_id, Some(0));
        assert!(records[2].parent_state_id.is_none());
    }

    #[test]
    fn test_read_summary() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("summary.parquet");

        let mut writer = TrajectoryWriter::new(path.clone());

        // 3 theorems: t1 (proved), t2 (not proved), t3 (proved)
        for i in 0..20 {
            let mut r = make_test_record(i, TrajectoryLabel::Positive);
            r.theorem_name = "t1".to_string();
            if i == 19 {
                r.is_proof_complete = true;
            }
            writer.record(r);
        }
        for i in 0..15 {
            let mut r = make_test_record(i, TrajectoryLabel::Negative);
            r.theorem_name = "t2".to_string();
            writer.record(r);
        }
        for i in 0..15 {
            let mut r = make_test_record(i, TrajectoryLabel::Positive);
            r.theorem_name = "t3".to_string();
            if i == 14 {
                r.is_proof_complete = true;
            }
            writer.record(r);
        }

        writer.finish().unwrap();

        let summary = TrajectoryReader::read_summary(&path).unwrap();
        assert_eq!(summary.total_records, 50);
        assert_eq!(summary.positive_count, 35); // 20 from t1 + 15 from t3
        assert_eq!(summary.negative_count, 15); // 15 from t2
        assert_eq!(summary.unique_theorems, 3);
        assert_eq!(summary.proved_theorems, 2); // t1 and t3
    }

    #[test]
    fn test_read_for_theorem() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("filter.parquet");

        let mut writer = TrajectoryWriter::new(path.clone());

        for i in 0..10 {
            let mut r = make_test_record(i, TrajectoryLabel::Positive);
            r.theorem_name = "theorem_a".to_string();
            writer.record(r);
        }
        for i in 0..5 {
            let mut r = make_test_record(i, TrajectoryLabel::Negative);
            r.theorem_name = "theorem_b".to_string();
            writer.record(r);
        }
        for i in 0..3 {
            let mut r = make_test_record(i, TrajectoryLabel::Unknown);
            r.theorem_name = "theorem_c".to_string();
            writer.record(r);
        }

        writer.finish().unwrap();

        let records_a = TrajectoryReader::read_for_theorem(&path, "theorem_a").unwrap();
        assert_eq!(records_a.len(), 10);
        assert!(records_a.iter().all(|r| r.theorem_name == "theorem_a"));

        let records_b = TrajectoryReader::read_for_theorem(&path, "theorem_b").unwrap();
        assert_eq!(records_b.len(), 5);

        let records_x = TrajectoryReader::read_for_theorem(&path, "nonexistent").unwrap();
        assert!(records_x.is_empty());
    }
}
