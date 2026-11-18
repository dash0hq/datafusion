// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Array caching utilities for scalar values

use std::cell::RefCell;
use std::collections::HashMap;
use std::iter::repeat_n;
use std::sync::{Arc, LazyLock, Mutex};

use arrow::array::{Array, ArrayRef, AsArray, PrimitiveArray, new_null_array};
use arrow::datatypes::{
    ArrowDictionaryKeyType, DataType, Fields, Int8Type, Int16Type, Int32Type, Int64Type,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};

use super::ScalarValue;

/// Maximum number of rows to cache to be conservative on memory usage
const MAX_CACHE_SIZE: usize = 1024 * 1024;

/// Cache for dictionary key arrays to avoid repeated allocations
/// when the same size is used frequently.
///
/// Similar to PartitionColumnProjector's ZeroBufferGenerators, this cache
/// stores key arrays for different dictionary key types. The cache is
/// limited to 1 entry per type (the last size used) to prevent memory leaks
/// for extremely large array requests.
#[derive(Debug)]
struct KeyArrayCache<K: ArrowDictionaryKeyType> {
    cache: Option<(usize, bool, PrimitiveArray<K>)>, // (num_rows, is_null, key_array)
}

impl<K: ArrowDictionaryKeyType> Default for KeyArrayCache<K> {
    fn default() -> Self {
        Self { cache: None }
    }
}

impl<K: ArrowDictionaryKeyType> KeyArrayCache<K> {
    /// Get or create a cached key array for the given number of rows and null status
    fn get_or_create(&mut self, num_rows: usize, is_null: bool) -> PrimitiveArray<K> {
        // Check cache size limit to prevent memory leaks
        if num_rows > MAX_CACHE_SIZE {
            // For very large arrays, don't cache them - just create and return
            return self.create_key_array(num_rows, is_null);
        }

        match &self.cache {
            Some((cached_num_rows, cached_is_null, cached_array))
                if *cached_num_rows == num_rows && *cached_is_null == is_null =>
            {
                // Cache hit: reuse existing array if same size and null status
                cached_array.clone()
            }
            _ => {
                // Cache miss: create new array and cache it
                let key_array = self.create_key_array(num_rows, is_null);
                self.cache = Some((num_rows, is_null, key_array.clone()));
                key_array
            }
        }
    }

    /// Create a new key array with the specified number of rows and null status
    fn create_key_array(&self, num_rows: usize, is_null: bool) -> PrimitiveArray<K> {
        let key_array: PrimitiveArray<K> = repeat_n(
            if is_null {
                None
            } else {
                Some(K::default_value())
            },
            num_rows,
        )
        .collect();
        key_array
    }
}

/// Cache for null arrays to avoid repeated allocations
/// when the same size is used frequently.
#[derive(Debug, Default)]
struct NullArrayCache {
    cache: Option<(usize, ArrayRef)>, // (num_rows, null_array)
}

impl NullArrayCache {
    /// Get or create a cached null array for the given number of rows
    fn get_or_create(&mut self, num_rows: usize) -> ArrayRef {
        // Check cache size limit to prevent memory leaks
        if num_rows > MAX_CACHE_SIZE {
            // For very large arrays, don't cache them - just create and return
            return new_null_array(&DataType::Null, num_rows);
        }

        match &self.cache {
            Some((cached_num_rows, cached_array)) if *cached_num_rows == num_rows => {
                // Cache hit: reuse existing array if same size
                Arc::clone(cached_array)
            }
            _ => {
                // Cache miss: create new array and cache it
                let null_array = new_null_array(&DataType::Null, num_rows);
                self.cache = Some((num_rows, Arc::clone(&null_array)));
                null_array
            }
        }
    }
}

/// Global cache for dictionary key arrays and null arrays
#[derive(Debug, Default)]
struct ArrayCaches {
    cache_i8: KeyArrayCache<Int8Type>,
    cache_i16: KeyArrayCache<Int16Type>,
    cache_i32: KeyArrayCache<Int32Type>,
    cache_i64: KeyArrayCache<Int64Type>,
    cache_u8: KeyArrayCache<UInt8Type>,
    cache_u16: KeyArrayCache<UInt16Type>,
    cache_u32: KeyArrayCache<UInt32Type>,
    cache_u64: KeyArrayCache<UInt64Type>,
    null_cache: NullArrayCache,
}

static ARRAY_CACHES: LazyLock<Mutex<ArrayCaches>> =
    LazyLock::new(|| Mutex::new(ArrayCaches::default()));

/// Get the global cache for arrays
fn get_array_caches() -> &'static Mutex<ArrayCaches> {
    &ARRAY_CACHES
}

/// Get or create a cached null array for the given number of rows
pub(crate) fn get_or_create_cached_null_array(num_rows: usize) -> ArrayRef {
    let cache = get_array_caches();
    let mut caches = cache.lock().unwrap();
    caches.null_cache.get_or_create(num_rows)
}

/// Get or create a cached key array for a specific key type
pub(crate) fn get_or_create_cached_key_array<K: ArrowDictionaryKeyType>(
    num_rows: usize,
    is_null: bool,
) -> PrimitiveArray<K> {
    let cache = get_array_caches();
    let mut caches = cache.lock().unwrap();

    // Use the DATA_TYPE to dispatch to the correct cache, similar to original implementation
    match K::DATA_TYPE {
        DataType::Int8 => {
            let array = caches.cache_i8.get_or_create(num_rows, is_null);
            // Convert using ArrayData to avoid unsafe transmute
            let array_data = array.to_data();
            PrimitiveArray::<K>::from(array_data)
        }
        DataType::Int16 => {
            let array = caches.cache_i16.get_or_create(num_rows, is_null);
            let array_data = array.to_data();
            PrimitiveArray::<K>::from(array_data)
        }
        DataType::Int32 => {
            let array = caches.cache_i32.get_or_create(num_rows, is_null);
            let array_data = array.to_data();
            PrimitiveArray::<K>::from(array_data)
        }
        DataType::Int64 => {
            let array = caches.cache_i64.get_or_create(num_rows, is_null);
            let array_data = array.to_data();
            PrimitiveArray::<K>::from(array_data)
        }
        DataType::UInt8 => {
            let array = caches.cache_u8.get_or_create(num_rows, is_null);
            let array_data = array.to_data();
            PrimitiveArray::<K>::from(array_data)
        }
        DataType::UInt16 => {
            let array = caches.cache_u16.get_or_create(num_rows, is_null);
            let array_data = array.to_data();
            PrimitiveArray::<K>::from(array_data)
        }
        DataType::UInt32 => {
            let array = caches.cache_u32.get_or_create(num_rows, is_null);
            let array_data = array.to_data();
            PrimitiveArray::<K>::from(array_data)
        }
        DataType::UInt64 => {
            let array = caches.cache_u64.get_or_create(num_rows, is_null);
            let array_data = array.to_data();
            PrimitiveArray::<K>::from(array_data)
        }
        _ => {
            // Fallback for unsupported types - create array directly without caching
            let key_array: PrimitiveArray<K> = repeat_n(
                if is_null {
                    None
                } else {
                    Some(K::default_value())
                },
                num_rows,
            )
            .collect();
            key_array
        }
    }
}

/// Narrowest struct whose typed null [`with_plan_scoped_nulls`] shares.
const SHARED_NULL_STRUCT_MIN_WIDTH: usize = 64;

tokio::task_local! {
    /// Typed null structs built in one physical planning call, keyed by the address of their
    /// `Fields`. Each entry owns that `Fields`, so the address cannot be reused while it lives.
    static PLAN_NULL_STRUCTS: RefCell<HashMap<usize, (Fields, ScalarValue)>>;
}

/// Runs `f` with a map that shares the typed null of each wide struct type for as long as `f`
/// runs. A typed null is immutable, so one value per `Fields` allocation serves every use.
pub async fn with_plan_scoped_nulls<F: Future>(f: F) -> F::Output {
    PLAN_NULL_STRUCTS
        .scope(RefCell::new(HashMap::new()), f)
        .await
}

pub(crate) fn plan_scoped_null_struct(fields: &Fields) -> ScalarValue {
    let build = || {
        ScalarValue::Struct(
            new_null_array(&DataType::Struct(fields.to_owned()), 1)
                .as_struct()
                .to_owned()
                .into(),
        )
    };
    if fields.len() < SHARED_NULL_STRUCT_MIN_WIDTH {
        return build();
    }
    let key = fields.as_ref().as_ptr() as usize;
    PLAN_NULL_STRUCTS
        .try_with(|map| {
            if let Some((_, value)) = map.borrow().get(&key) {
                return value.clone();
            }
            let value = build();
            map.borrow_mut()
                .insert(key, (fields.clone(), value.clone()));
            value
        })
        .unwrap_or_else(|_| build())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::Field;
    use std::pin::pin;
    use std::task::{Context, Poll, Waker};

    #[test]
    fn plan_scoped_nulls_are_shared_within_one_call_only() {
        let struct_type = DataType::Struct(
            (0..SHARED_NULL_STRUCT_MIN_WIDTH)
                .map(|i| Field::new(format!("c{i}"), DataType::Int32, true))
                .collect(),
        );
        let null = || match ScalarValue::try_new_null(&struct_type).unwrap() {
            ScalarValue::Struct(array) => array,
            other => panic!("expected a struct null, got {other:?}"),
        };
        let call = with_plan_scoped_nulls(async {
            let first = null();
            assert!(Arc::ptr_eq(&first, &null()));
            Arc::downgrade(&first)
        });
        let Poll::Ready(shared) =
            pin!(call).poll(&mut Context::from_waker(Waker::noop()))
        else {
            unreachable!("the call never awaits");
        };
        assert!(shared.upgrade().is_none());
        assert!(!Arc::ptr_eq(&null(), &null()));
    }
}
