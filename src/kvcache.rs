use std::{usize, vec};

use crate::tensor::Tensor;

// KVCache 结构体，用于存储键和值的缓存
pub struct KVCache<T> {
    k_cache: Vec<Tensor<T>>, // 键缓存 (max_seq_len, n_kv_head * dqkv) x layers
    v_cache: Vec<Tensor<T>>, // 值缓存 (max_seq_len, n_kv_head * dqkv) x layers
    #[allow(unused)]
    max_seq_len: usize, // 最大序列长度
    dim: usize,              // 维度
    length: usize,           // 当前序列的长度
}

impl<T: Default + Copy> KVCache<T> {
    // 创建一个新的 KVCache 实例
    pub fn new(n_layers: usize, max_seq_len: usize, dim: usize, init_len: usize) -> Self {
        KVCache {
            k_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(), // 初始化键缓存
            v_cache: (0..n_layers)
                .map(|_| Tensor::default(&vec![max_seq_len, dim]))
                .collect(), // 初始化值缓存
            max_seq_len: max_seq_len,
            dim: dim,
            length: init_len,
        }
    }

    // 获取指定层和起始位置的键缓存切片
    pub fn k_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.k_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    // 获取指定层和起始位置的值缓存切片
    pub fn v_cache(&mut self, layer: usize, start: usize) -> Tensor<T> {
        self.v_cache[layer].slice(start * self.dim, &vec![self.length - start, self.dim])
    }

    // 增加序列长度
    pub fn increment(&mut self, seq_len: usize) {
        self.length += seq_len;
    }

    // 获取当前序列的长度
    pub fn len(&self) -> usize {
        self.length
    }
}
