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
    ///
    /// Streams batches and only extracts the columns needed for summary
    /// computation, avoiding full record deserialization.
    pub fn read_summary(path: &Path) -> anyhow::Result<TrajectorySummary> {
        let file = std::fs::File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        let mut total_records = 0;
        let mut positive_count = 0;
        let mut negative_count = 0;
        let mut theorem_names = HashSet::new();
        let mut proved_theorems = HashSet::new();

        for batch_result in reader {
            let batch = batch_result?;
            let num_rows = batch.num_rows();
            total_records += num_rows;

            let names = get_string_column(&batch, "theorem_name")?;
            let labels = get_string_column(&batch, "label")?;
            let proof_complete = get_bool_column(&batch, "is_proof_complete")?;

            for i in 0..num_rows {
                match TrajectoryLabel::from_str_lossy(labels.value(i)) {
                    TrajectoryLabel::Positive => positive_count += 1,
                    TrajectoryLabel::Negative => negative_count += 1,
                    TrajectoryLabel::Unknown => {}
                }
                theorem_names.insert(names.value(i).to_string());
                if proof_complete.value(i) {
                    proved_theorems.insert(names.value(i).to_string());
                }
            }
        }

        Ok(TrajectorySummary {
            total_records,
            positive_count,
            negative_count,
            unique_theorems: theorem_names.len(),
            proved_theorems: proved_theorems.len(),
        })
    }

    /// Read unique theorem names from a Parquet file (for resume filtering).
    ///
    /// Streams batches and only extracts the `theorem_name` column.
    pub fn read_theorem_names(path: &Path) -> anyhow::Result<HashSet<String>> {
        let file = std::fs::File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        let mut names = HashSet::new();
        for batch_result in reader {
            let batch = batch_result?;
            let theorem_names = get_string_column(&batch, "theorem_name")?;
            for i in 0..batch.num_rows() {
                names.insert(theorem_names.value(i).to_string());
            }
        }

        Ok(names)
    }

    /// Read only records for a specific theorem from a Parquet file.
    ///
    /// Streams batches and only deserializes rows where `theorem_name` matches.
    pub fn read_for_theorem(path: &Path, theorem_name: &str) -> anyhow::Result<Vec<TrajectoryRecord>> {
        let file = std::fs::File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        let mut records = Vec::new();
        for batch_result in reader {
            let batch = batch_result?;
            let names = get_string_column(&batch, "theorem_name")?;

            let has_match = (0..batch.num_rows()).any(|i| names.value(i) == theorem_name);
            if !has_match {
                continue;
            }

            // Only deserialize the full batch when it contains matching rows
            let all_batch_records = extract_records_from_batch(&batch)?;
            for record in all_batch_records {
                if record.theorem_name == theorem_name {
                    records.push(record);
                }
            }
        }

        Ok(records)
    }
}

/// Helper to get a named StringArray column from a RecordBatch.
fn get_string_column<'a>(batch: &'a RecordBatch, name: &str) -> anyhow::Result<&'a StringArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("Missing column: {name}"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow::anyhow!("Column {name} is not StringArray"))
}

/// Helper to get a named BooleanArray column from a RecordBatch.
fn get_bool_column<'a>(batch: &'a RecordBatch, name: &str) -> anyhow::Result<&'a BooleanArray> {
    batch
        .column_by_name(name)
        .ok_or_else(|| anyhow::anyhow!("Missing column: {name}"))?
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| anyhow::anyhow!("Column {name} is not BooleanArray"))
}

/// Extract trajectory records from a single Arrow RecordBatch using named columns.
fn extract_records_from_batch(batch: &RecordBatch) -> anyhow::Result<Vec<TrajectoryRecord>> {
    let theorem_names = get_string_column(batch, "theorem_name")?;

    let state_ids = batch
        .column_by_name("state_id")
        .ok_or_else(|| anyhow::anyhow!("Missing column: state_id"))?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| anyhow::anyhow!("Column state_id is not UInt64Array"))?;

    let state_pps = get_string_column(batch, "state_pp")?;
    let tactics = get_string_column(batch, "tactic_applied")?;

    let parent_ids = batch
        .column_by_name("parent_state_id")
        .ok_or_else(|| anyhow::anyhow!("Missing column: parent_state_id"))?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| anyhow::anyhow!("Column parent_state_id is not UInt64Array"))?;

    let labels = get_string_column(batch, "label")?;

    let depths = batch
        .column_by_name("depth_from_root")
        .ok_or_else(|| anyhow::anyhow!("Missing column: depth_from_root"))?
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| anyhow::anyhow!("Column depth_from_root is not UInt32Array"))?;

    let remaining = batch
        .column_by_name("remaining_depth")
        .ok_or_else(|| anyhow::anyhow!("Missing column: remaining_depth"))?
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| anyhow::anyhow!("Column remaining_depth is not Int32Array"))?;

    let log_probs = batch
        .column_by_name("llm_log_prob")
        .ok_or_else(|| anyhow::anyhow!("Missing column: llm_log_prob"))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| anyhow::anyhow!("Column llm_log_prob is not Float64Array"))?;

    let ebm_scores = batch
        .column_by_name("ebm_score")
        .ok_or_else(|| anyhow::anyhow!("Missing column: ebm_score"))?
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| anyhow::anyhow!("Column ebm_score is not Float64Array"))?;

    let proof_complete = get_bool_column(batch, "is_proof_complete")?;

    let timestamps = batch
        .column_by_name("timestamp_ms")
        .ok_or_else(|| anyhow::anyhow!("Missing column: timestamp_ms"))?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| anyhow::anyhow!("Column timestamp_ms is not UInt64Array"))?;

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

    #[test]
    fn test_read_theorem_names() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("names.parquet");

        let mut writer = TrajectoryWriter::new(path.clone());
        for i in 0..5 {
            let mut r = make_test_record(i, TrajectoryLabel::Positive);
            r.theorem_name = "alpha".to_string();
            writer.record(r);
        }
        for i in 0..3 {
            let mut r = make_test_record(i, TrajectoryLabel::Negative);
            r.theorem_name = "beta".to_string();
            writer.record(r);
        }
        for i in 0..2 {
            let mut r = make_test_record(i, TrajectoryLabel::Unknown);
            r.theorem_name = "gamma".to_string();
            writer.record(r);
        }
        writer.finish().unwrap();

        let names = TrajectoryReader::read_theorem_names(&path).unwrap();
        assert_eq!(names.len(), 3);
        assert!(names.contains("alpha"));
        assert!(names.contains("beta"));
        assert!(names.contains("gamma"));
    }
}
