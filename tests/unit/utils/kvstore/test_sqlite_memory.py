# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest

from llama_stack.providers.utils.kvstore.sqlite.config import SqliteKVStoreConfig
from llama_stack.providers.utils.kvstore.sqlite.sqlite import SqliteKVStoreImpl


async def test_memory_kvstore_basic_operations():
    """Test basic CRUD operations with :memory: database."""
    config = SqliteKVStoreConfig(db_path=":memory:")
    store = SqliteKVStoreImpl(config)
    await store.initialize()

    # Test set and get
    await store.set("key1", "value1")
    result = await store.get("key1")
    assert result == "value1"

    # Test get non-existent key
    result = await store.get("nonexistent")
    assert result is None

    # Test update
    await store.set("key1", "updated_value")
    result = await store.get("key1")
    assert result == "updated_value"

    # Test delete
    await store.delete("key1")
    result = await store.get("key1")
    assert result is None

    await store.close()


async def test_memory_kvstore_range_operations():
    """Test range query operations with :memory: database."""
    config = SqliteKVStoreConfig(db_path=":memory:")
    store = SqliteKVStoreImpl(config)
    await store.initialize()

    # Set up test data
    await store.set("key_a", "value_a")
    await store.set("key_b", "value_b")
    await store.set("key_c", "value_c")
    await store.set("key_d", "value_d")

    # Test values_in_range
    values = await store.values_in_range("key_b", "key_c")
    assert len(values) == 2
    assert "value_b" in values
    assert "value_c" in values

    # Test keys_in_range
    keys = await store.keys_in_range("key_a", "key_c")
    assert len(keys) == 3
    assert "key_a" in keys
    assert "key_b" in keys
    assert "key_c" in keys

    await store.close()


async def test_memory_kvstore_multiple_instances():
    """Test that multiple :memory: instances are independent."""
    config1 = SqliteKVStoreConfig(db_path=":memory:")
    config2 = SqliteKVStoreConfig(db_path=":memory:")

    store1 = SqliteKVStoreImpl(config1)
    store2 = SqliteKVStoreImpl(config2)

    await store1.initialize()
    await store2.initialize()

    # Set data in store1
    await store1.set("shared_key", "value_from_store1")

    # Verify store2 doesn't see store1's data
    result = await store2.get("shared_key")
    assert result is None

    # Set different value in store2
    await store2.set("shared_key", "value_from_store2")

    # Verify both stores have independent data
    assert await store1.get("shared_key") == "value_from_store1"
    assert await store2.get("shared_key") == "value_from_store2"

    await store1.close()
    await store2.close()


async def test_memory_kvstore_persistence_behavior():
    """Test that :memory: database doesn't persist across instances."""
    config = SqliteKVStoreConfig(db_path=":memory:")

    # First instance
    store1 = SqliteKVStoreImpl(config)
    await store1.initialize()
    await store1.set("persist_test", "should_not_persist")
    await store1.close()

    # Second instance with same config
    store2 = SqliteKVStoreImpl(config)
    await store2.initialize()

    # Data should not be present
    result = await store2.get("persist_test")
    assert result is None

    await store2.close()
