use serde; // 引入serde库用于序列化和反序列化

// 定义LlamaConfigJson结构体，并派生Serialize、Deserialize和Debug特性
#[derive(serde::Serialize, serde::Deserialize, Debug)]
pub(crate) struct LlamaConfigJson {
    pub bos_token_id: u32,              // 开始标记ID
    pub eos_token_id: u32,              // 结束标记ID
    pub hidden_size: usize,             // 隐藏层大小
    pub intermediate_size: usize,       // 中间层大小
    pub max_position_embeddings: usize, // 最大位置嵌入数
    pub num_attention_heads: usize,     // 注意力头数量
    pub num_hidden_layers: usize,       // 隐藏层数量
    pub num_key_value_heads: usize,     // 键值头数量
    pub vocab_size: usize,              // 词汇表大小
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32, // RMS归一化的epsilon值，默认值为1e-5
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32, // ROPE的theta值，默认值为1e4
    pub torch_dtype: String,            // Torch的数据类型
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool, // 是否绑定词嵌入，默认值为false
}

// 定义默认的RMS归一化epsilon值
#[inline(always)]
const fn default_rms_norm_eps() -> f32 {
    1e-5
}

// 定义默认的ROPE theta值
#[inline(always)]
const fn default_rope_theta() -> f32 {
    1e4
}

// 定义默认的词嵌入绑定值
#[inline(always)]
const fn default_tie_word_embeddings() -> bool {
    false
}
