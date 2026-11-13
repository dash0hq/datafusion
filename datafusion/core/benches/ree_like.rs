// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use arrow::{
    array::{ArrayRef, Int32Array, RunArray, StringArray},
    datatypes::{Field, Int32Type, Schema},
    record_batch::RecordBatch,
};
use criterion::{Criterion, criterion_group, criterion_main};
use datafusion::prelude::SessionContext;
use datafusion::{datasource::MemTable, error::Result};
use std::hint::black_box;
use std::sync::Arc;
use tokio::runtime::Runtime;

fn query(ctx: &SessionContext, rt: &Runtime, sql: &str) {
    let df = rt.block_on(ctx.sql(sql)).unwrap();
    black_box(rt.block_on(df.collect()).unwrap());
}

/// Each batch holds `batch_size / run_length` distinct strings, each repeated
/// `run_length` times. Half of the strings match `LIKE '%foo%'`.
fn create_context(
    array_len: usize,
    batch_size: usize,
    run_length: usize,
    run_end_encoded: bool,
) -> Result<SessionContext> {
    let num_runs = batch_size / run_length;
    let run_values: Vec<String> = (0..num_runs)
        .map(|i| {
            if i % 2 == 0 {
                format!("foo_{i}")
            } else {
                format!("bar_{i}")
            }
        })
        .collect();

    let column: ArrayRef = if run_end_encoded {
        let run_ends: Int32Array =
            (1..=num_runs).map(|i| (i * run_length) as i32).collect();
        let values = StringArray::from(run_values);
        Arc::new(RunArray::<Int32Type>::try_new(&run_ends, &values)?)
    } else {
        Arc::new(StringArray::from_iter_values(
            (0..batch_size).map(|i| &run_values[i / run_length]),
        ))
    };

    let schema = Arc::new(Schema::new(vec![Field::new(
        "s",
        column.data_type().clone(),
        false,
    )]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![column])?;
    let batches = vec![batch; array_len / batch_size];

    let ctx = SessionContext::new();
    let provider = MemTable::try_new(schema, vec![batches])?;
    ctx.register_table("t", Arc::new(provider))?;

    Ok(ctx)
}

fn criterion_benchmark(c: &mut Criterion) {
    let array_len = 524_288; // 2^19
    let batch_size = 8192; // 2^13
    let run_length = 64;
    let rt = Runtime::new().unwrap();
    let sql = "select count(*) from t where s like '%foo%'";

    c.bench_function("like_utf8", |b| {
        let ctx = create_context(array_len, batch_size, run_length, false).unwrap();
        b.iter(|| query(&ctx, &rt, sql))
    });

    c.bench_function("like_ree_utf8", |b| {
        let ctx = create_context(array_len, batch_size, run_length, true).unwrap();
        b.iter(|| query(&ctx, &rt, sql))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
