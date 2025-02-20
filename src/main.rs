mod config; // 引入配置模块
mod kvcache; // 引入键值缓存模块
mod model; // 引入模型模块
mod operators; // 引入操作符模块
mod params; // 引入参数模块
mod tensor; // 引入张量模块

use std::path::PathBuf; // 引入PathBuf用于处理文件路径
use tokenizers::Tokenizer; // 引入Tokenizer用于处理文本标记化

fn main() {
    let project_dir = env!("CARGO_MANIFEST_DIR"); // 获取项目目录
    let model_dir = PathBuf::from(project_dir).join("models").join("story"); // 构建模型目录路径
    let llama = model::Llama::<f32>::from_safetensors(&model_dir); // 从模型目录加载Llama模型
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap(); // 从文件加载标记器
    let input = "Once upon a time"; // 输入文本
    let binding = tokenizer.encode(input, true).unwrap(); // 对输入文本进行编码
    let input_ids = binding.get_ids(); // 获取编码后的ID
    print!("\n{}", input); // 打印输入文本
    let output_ids = llama.generate(
        input_ids, // 输入ID
        500,       // 生成的最大标记数
        0.8,       // 温度参数
        30,        // 顶部P参数
        1.,        // 顶部K参数
    );
    println!("{}", tokenizer.decode(&output_ids, true).unwrap()); // 解码并打印生成的文本
}
