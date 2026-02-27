//! Writes TrajectoryRecords to Parquet files using Arrow.

use crate::reader::TrajectoryReader;
use crate::types::{SearchResult, TrajectoryLabel, TrajectoryRecord};
use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

/// Arrow schema for trajectory Parquet files (14 columns).
pub fn trajectory_schema() -> Schema {
    Schema::new(vec![
        Field::new("theorem_name", DataType::Utf8, false),
        Field::new("state_id", DataType::UInt64, false),
        Field::new("state_pp", DataType::Utf8, false),
        Field::new("tactic_applied", DataType::Utf8, false),
        Field::new("parent_state_id", DataType::UInt64, true),
        Field::new("label", DataType::Utf8, false),
        Field::new("depth_from_root", DataType::UInt32, false),
        Field::new("remaining_depth", DataType::Int32, false),
        Field::new("llm_log_prob", DataType::Float64, false),
        Field::new("ebm_score", DataType::Float64, false),
        Field::new("is_proof_complete", DataType::Boolean, false),
        Field::new("timestamp_ms", DataType::UInt64, false),
        Field::new("q_value", DataType::Float64, false),
        Field::new("visits", DataType::UInt32, false),
    ])
}

/// Buffers trajectory records and writes them to a Parquet file.
///
/// Supports periodic checkpointing via [`flush_partial`](TrajectoryWriter::flush_partial)
/// for crash resilience during long search runs.
pub struct TrajectoryWriter {
    /// Records not yet written to disk.
    pending: Vec<TrajectoryRecord>,
    /// Number of records already flushed to disk.
    flushed_count: usize,
    output_path: PathBuf,
}

impl TrajectoryWriter {
    /// Create a new writer that will write to the given path.
    pub fn new(output_path: PathBuf) -> Self {
        Self {
            pending: Vec::new(),
            flushed_count: 0,
            output_path,
        }
    }

    /// Buffer a single trajectory record.
    pub fn record(&mut self, record: TrajectoryRecord) {
        self.pending.push(record);
    }

    /// Buffer multiple trajectory records.
    pub fn record_all(&mut self, records: Vec<TrajectoryRecord>) {
        self.pending.extend(records);
    }

    /// Total number of records (both flushed and pending).
    pub fn len(&self) -> usize {
        self.flushed_count + self.pending.len()
    }

    /// Whether both buffers are empty.
    pub fn is_empty(&self) -> bool {
        self.flushed_count == 0 && self.pending.is_empty()
    }

    /// Write pending records to disk as a checkpoint without consuming self.
    ///
    /// Reads back any previously flushed records from the output file,
    /// combines them with pending records, and rewrites the file. This
    /// avoids accumulating all flushed records in memory.
    pub fn flush_partial(&mut self) -> anyhow::Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }

        // Read back previously flushed records from disk (if any)
        let mut all = if self.flushed_count > 0 && self.output_path.exists() {
            TrajectoryReader::read_all(&self.output_path)?
        } else {
            Vec::new()
        };

        all.append(&mut self.pending);
        self.flushed_count = all.len();

        let batch = build_record_batch(&all)?;
        let schema = Arc::new(trajectory_schema());
        let file = std::fs::File::create(&self.output_path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;

        tracing::debug!(
            records = self.flushed_count,
            path = %self.output_path.display(),
            "Flushed partial checkpoint"
        );

        Ok(())
    }

    /// Write all records (flushed + pending) to the Parquet file and return the output path.
    pub fn finish(mut self) -> anyhow::Result<PathBuf> {
        // Read back previously flushed records from disk (if any)
        let mut all = if self.flushed_count > 0 && self.output_path.exists() {
            TrajectoryReader::read_all(&self.output_path)?
        } else {
            Vec::new()
        };

        all.append(&mut self.pending);
        let schema = Arc::new(trajectory_schema());

        let batch = if all.is_empty() {
            RecordBatch::new_empty(schema.clone())
        } else {
            build_record_batch(&all)?
        };

        let file = std::fs::File::create(&self.output_path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;

        tracing::info!(
            records = all.len(),
            path = %self.output_path.display(),
            "Wrote trajectory Parquet file"
        );

        Ok(self.output_path)
    }

    /// Assign labels and remaining_depth to records from a search result.
    ///
    /// If the theorem was proved, traces the proof path from the terminal node
    /// back to the root and labels path nodes as Positive with correct
    /// remaining_depth. Off-path nodes are labeled Negative with remaining_depth=-1.
    ///
    /// If the theorem was not proved, all records are labeled Negative.
    pub fn from_search_result(result: &SearchResult) -> Vec<TrajectoryRecord> {
        let mut records = result.all_records.clone();

        if result.proved {
            // Find the terminal node (is_proof_complete == true)
            let terminal = records.iter().find(|r| r.is_proof_complete);

            if let Some(terminal) = terminal {
                // Trace backward from terminal to root using parent_state_id
                let mut proof_path_ids = Vec::new();
                let mut current_id = Some(terminal.state_id);

                // Build a map from state_id -> record for fast lookup
                let id_to_record: std::collections::HashMap<u64, &TrajectoryRecord> =
                    records.iter().map(|r| (r.state_id, r)).collect();

                while let Some(id) = current_id {
                    proof_path_ids.push(id);
                    if let Some(record) = id_to_record.get(&id) {
                        current_id = record.parent_state_id;
                    } else {
                        break;
                    }
                }

                // Reverse so proof_path_ids goes root → terminal
                proof_path_ids.reverse();
                let proof_path_set: HashSet<u64> = proof_path_ids.iter().copied().collect();
                let path_len = proof_path_ids.len();

                // Build index map: state_id → position in proof path
                let path_index: std::collections::HashMap<u64, usize> = proof_path_ids
                    .iter()
                    .enumerate()
                    .map(|(i, &id)| (id, i))
                    .collect();

                for record in &mut records {
                    if proof_path_set.contains(&record.state_id) {
                        record.label = TrajectoryLabel::Positive;
                        let idx = path_index[&record.state_id];
                        record.remaining_depth = (path_len - 1 - idx) as i32;
                    } else {
                        record.label = TrajectoryLabel::Negative;
                        record.remaining_depth = -1;
                    }
                }
            } else {
                // Marked as proved but no terminal node found — label all negative
                for record in &mut records {
                    record.label = TrajectoryLabel::Negative;
                    record.remaining_depth = -1;
                }
            }
        } else {
            // Not proved — all negative
            for record in &mut records {
                record.label = TrajectoryLabel::Negative;
                record.remaining_depth = -1;
            }
        }

        records
    }
}

/// Build an Arrow RecordBatch from trajectory records.
fn build_record_batch(records: &[TrajectoryRecord]) -> anyhow::Result<RecordBatch> {
    let schema = Arc::new(trajectory_schema());

    let theorem_names: StringArray = records.iter().map(|r| Some(r.theorem_name.as_str())).collect();
    let state_ids: UInt64Array = records.iter().map(|r| Some(r.state_id)).collect();
    let state_pps: StringArray = records.iter().map(|r| Some(r.state_pp.as_str())).collect();
    let tactics: StringArray = records.iter().map(|r| Some(r.tactic_applied.as_str())).collect();

    let mut parent_ids_builder = UInt64Builder::new();
    for r in records {
        match r.parent_state_id {
            Some(v) => parent_ids_builder.append_value(v),
            None => parent_ids_builder.append_null(),
        }
    }
    let parent_ids = parent_ids_builder.finish();

    let labels: StringArray = records.iter().map(|r| Some(r.label.to_string())).collect();
    let depths: UInt32Array = records.iter().map(|r| Some(r.depth_from_root)).collect();
    let remaining: Int32Array = records.iter().map(|r| Some(r.remaining_depth)).collect();
    let log_probs: Float64Array = records.iter().map(|r| Some(r.llm_log_prob)).collect();
    let ebm_scores: Float64Array = records.iter().map(|r| Some(r.ebm_score)).collect();
    let proof_complete: BooleanArray = records.iter().map(|r| Some(r.is_proof_complete)).collect();
    let timestamps: UInt64Array = records.iter().map(|r| Some(r.timestamp_ms)).collect();
    let q_values: Float64Array = records.iter().map(|r| Some(r.q_value)).collect();
    let visits: UInt32Array = records.iter().map(|r| Some(r.visits)).collect();

    let columns: Vec<Arc<dyn arrow::array::Array>> = vec![
        Arc::new(theorem_names),
        Arc::new(state_ids),
        Arc::new(state_pps),
        Arc::new(tactics),
        Arc::new(parent_ids),
        Arc::new(labels),
        Arc::new(depths),
        Arc::new(remaining),
        Arc::new(log_probs),
        Arc::new(ebm_scores),
        Arc::new(proof_complete),
        Arc::new(timestamps),
        Arc::new(q_values),
        Arc::new(visits),
    ];

    Ok(RecordBatch::try_new(schema, columns)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SearchStats, TrajectoryLabel};
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
            q_value: 0.0,
            visits: 0,
        }
    }

    #[test]
    fn test_trajectory_schema_has_14_columns() {
        let schema = trajectory_schema();
        assert_eq!(schema.fields().len(), 14);
        assert_eq!(schema.field(0).name(), "theorem_name");
        assert_eq!(schema.field(4).name(), "parent_state_id");
        assert!(schema.field(4).is_nullable());
        assert_eq!(schema.field(11).name(), "timestamp_ms");
    }

    #[test]
    fn test_write_empty_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("empty.parquet");
        let writer = TrajectoryWriter::new(path.clone());
        assert_eq!(writer.len(), 0);
        let result = writer.finish().unwrap();
        assert_eq!(result, path);
        assert!(path.exists());
    }

    #[test]
    fn test_write_and_verify_file_exists() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.parquet");
        let mut writer = TrajectoryWriter::new(path.clone());

        for i in 0..10 {
            writer.record(make_test_record(i, TrajectoryLabel::Unknown));
        }
        assert_eq!(writer.len(), 10);

        let result = writer.finish().unwrap();
        assert!(result.exists());
        assert!(std::fs::metadata(&result).unwrap().len() > 0);
    }

    #[test]
    fn test_from_search_result_proved() {
        // Tree structure:
        //   0 (root) → 1 → 2 → 3 (QED)
        //                ↘ 4 (dead end)
        let mut root = make_test_record(0, TrajectoryLabel::Unknown);
        root.parent_state_id = None;

        let mut n1 = make_test_record(1, TrajectoryLabel::Unknown);
        n1.parent_state_id = Some(0);

        let mut n2 = make_test_record(2, TrajectoryLabel::Unknown);
        n2.parent_state_id = Some(1);

        let mut n3 = make_test_record(3, TrajectoryLabel::Unknown);
        n3.parent_state_id = Some(2);
        n3.is_proof_complete = true;

        let mut n4 = make_test_record(4, TrajectoryLabel::Unknown);
        n4.parent_state_id = Some(1);

        let result = SearchResult {
            theorem_name: "test_theorem".to_string(),
            proved: true,
            proof_tactics: vec!["tactic_1".into(), "tactic_2".into(), "tactic_3".into()],
            nodes_expanded: 5,
            total_states: 5,
            max_depth_reached: 3,
            wall_time_ms: 1000,
            all_records: vec![root, n1, n2, n3, n4],
            stats: SearchStats::default(),
            failure_reason: String::new(),
        };

        let labeled = TrajectoryWriter::from_search_result(&result);
        assert_eq!(labeled.len(), 5);

        // Proof path: 0 → 1 → 2 → 3 (4 nodes, remaining_depth: 3, 2, 1, 0)
        let by_id: std::collections::HashMap<u64, &TrajectoryRecord> =
            labeled.iter().map(|r| (r.state_id, r)).collect();

        assert_eq!(by_id[&0].label, TrajectoryLabel::Positive);
        assert_eq!(by_id[&0].remaining_depth, 3);

        assert_eq!(by_id[&1].label, TrajectoryLabel::Positive);
        assert_eq!(by_id[&1].remaining_depth, 2);

        assert_eq!(by_id[&2].label, TrajectoryLabel::Positive);
        assert_eq!(by_id[&2].remaining_depth, 1);

        assert_eq!(by_id[&3].label, TrajectoryLabel::Positive);
        assert_eq!(by_id[&3].remaining_depth, 0);

        assert_eq!(by_id[&4].label, TrajectoryLabel::Negative);
        assert_eq!(by_id[&4].remaining_depth, -1);
    }

    #[test]
    fn test_from_search_result_single_node_proof() {
        // Root node IS the terminal (state_id=0, is_proof_complete=true, parent=None).
        // Exercises the path-tracing loop with a single-element path.
        let mut root = make_test_record(0, TrajectoryLabel::Unknown);
        root.parent_state_id = None;
        root.is_proof_complete = true;

        let result = SearchResult {
            theorem_name: "test_theorem".to_string(),
            proved: true,
            proof_tactics: vec![],
            nodes_expanded: 0,
            total_states: 1,
            max_depth_reached: 0,
            wall_time_ms: 10,
            all_records: vec![root],
            stats: SearchStats::default(),
            failure_reason: String::new(),
        };

        let labeled = TrajectoryWriter::from_search_result(&result);
        assert_eq!(labeled.len(), 1);
        assert_eq!(labeled[0].label, TrajectoryLabel::Positive);
        assert_eq!(labeled[0].remaining_depth, 0);
    }

    #[test]
    fn test_from_search_result_proved_but_no_terminal() {
        // proved=true but no record has is_proof_complete=true.
        // Hits the defensive fallback (else branch at writer.rs:143-149)
        // which labels everything Negative.
        let records: Vec<TrajectoryRecord> = (0..3)
            .map(|i| make_test_record(i, TrajectoryLabel::Unknown))
            .collect();

        let result = SearchResult {
            theorem_name: "test_theorem".to_string(),
            proved: true,
            proof_tactics: vec!["tactic_1".into(), "tactic_2".into()],
            nodes_expanded: 3,
            total_states: 3,
            max_depth_reached: 2,
            wall_time_ms: 500,
            all_records: records,
            stats: SearchStats::default(),
            failure_reason: String::new(),
        };

        let labeled = TrajectoryWriter::from_search_result(&result);
        assert_eq!(labeled.len(), 3);
        for record in &labeled {
            assert_eq!(record.label, TrajectoryLabel::Negative);
            assert_eq!(record.remaining_depth, -1);
        }
    }

    #[test]
    fn test_from_search_result_multiple_terminals() {
        // Two proof paths: root(0) → 1 → 2(QED) and root(0) → 3 → 4(QED).
        // from_search_result() uses .find() which returns the first terminal (2).
        // Path 0→1→2 should be Positive; records 3 and 4 should be Negative.
        let mut root = make_test_record(0, TrajectoryLabel::Unknown);
        root.parent_state_id = None;

        let mut n1 = make_test_record(1, TrajectoryLabel::Unknown);
        n1.parent_state_id = Some(0);

        let mut n2 = make_test_record(2, TrajectoryLabel::Unknown);
        n2.parent_state_id = Some(1);
        n2.is_proof_complete = true;

        let mut n3 = make_test_record(3, TrajectoryLabel::Unknown);
        n3.parent_state_id = Some(0);

        let mut n4 = make_test_record(4, TrajectoryLabel::Unknown);
        n4.parent_state_id = Some(3);
        n4.is_proof_complete = true;

        let result = SearchResult {
            theorem_name: "test_theorem".to_string(),
            proved: true,
            proof_tactics: vec!["tactic_1".into(), "tactic_2".into()],
            nodes_expanded: 5,
            total_states: 5,
            max_depth_reached: 2,
            wall_time_ms: 1000,
            all_records: vec![root, n1, n2, n3, n4],
            stats: SearchStats::default(),
            failure_reason: String::new(),
        };

        let labeled = TrajectoryWriter::from_search_result(&result);
        assert_eq!(labeled.len(), 5);

        let by_id: std::collections::HashMap<u64, &TrajectoryRecord> =
            labeled.iter().map(|r| (r.state_id, r)).collect();

        // Path 0→1→2 is Positive
        assert_eq!(by_id[&0].label, TrajectoryLabel::Positive);
        assert_eq!(by_id[&0].remaining_depth, 2);
        assert_eq!(by_id[&1].label, TrajectoryLabel::Positive);
        assert_eq!(by_id[&1].remaining_depth, 1);
        assert_eq!(by_id[&2].label, TrajectoryLabel::Positive);
        assert_eq!(by_id[&2].remaining_depth, 0);

        // Records 3 and 4 are off-path → Negative
        assert_eq!(by_id[&3].label, TrajectoryLabel::Negative);
        assert_eq!(by_id[&3].remaining_depth, -1);
        assert_eq!(by_id[&4].label, TrajectoryLabel::Negative);
        assert_eq!(by_id[&4].remaining_depth, -1);
    }

    #[test]
    fn test_from_search_result_unproved() {
        let records: Vec<TrajectoryRecord> = (0..5)
            .map(|i| make_test_record(i, TrajectoryLabel::Unknown))
            .collect();

        let result = SearchResult {
            theorem_name: "test_theorem".to_string(),
            proved: false,
            proof_tactics: vec![],
            nodes_expanded: 5,
            total_states: 5,
            max_depth_reached: 4,
            wall_time_ms: 5000,
            all_records: records,
            stats: SearchStats::default(),
            failure_reason: String::new(),
        };

        let labeled = TrajectoryWriter::from_search_result(&result);
        for record in &labeled {
            assert_eq!(record.label, TrajectoryLabel::Negative);
            assert_eq!(record.remaining_depth, -1);
        }
    }

    #[test]
    fn test_flush_partial_writes_checkpoint() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("partial.parquet");

        let mut writer = TrajectoryWriter::new(path.clone());

        // Add 3 records and flush
        for i in 0..3 {
            writer.record(make_test_record(i, TrajectoryLabel::Positive));
        }
        writer.flush_partial().unwrap();
        assert_eq!(writer.len(), 3); // 3 flushed, 0 pending

        // Checkpoint file should exist and be readable
        let mid_records = crate::reader::TrajectoryReader::read_all(&path).unwrap();
        assert_eq!(mid_records.len(), 3);

        // Add 2 more records and finish
        for i in 3..5 {
            writer.record(make_test_record(i, TrajectoryLabel::Negative));
        }
        assert_eq!(writer.len(), 5); // 3 flushed + 2 pending

        writer.finish().unwrap();

        // Final file should have all 5 records
        let final_records = crate::reader::TrajectoryReader::read_all(&path).unwrap();
        assert_eq!(final_records.len(), 5);
    }

    #[test]
    fn test_flush_partial_empty_noop() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("noop.parquet");

        let mut writer = TrajectoryWriter::new(path.clone());
        // Flush with no records should not create a file
        writer.flush_partial().unwrap();
        assert!(!path.exists());
    }
}
