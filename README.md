# Image Matting

一个基于 Rust 的图像抠图工具，使用 BiRefNet 模型实现高质量的图像抠图功能。

## 功能特点

- 支持图像抠图（去除背景）
- 支持生成掩码（alpha 通道）
- 提供 RESTful API 接口
- 包含简单的前端界面

## 技术栈

- 后端：Rust + Rocket
- 前端：React + TypeScript
- 模型：BiRefNet
- 推理引擎：ONNX Runtime

## 环境要求

- Rust 1.70+
- Node.js 16+
- npm 8+

## 安装

1. 克隆项目：
```bash
git clone https://github.com/jkin8010/image-matting.git
cd image-matting
```

2. 安装依赖：
```bash
# 安装 Rust 依赖
cargo build

# 安装前端依赖
cd frontend
npm install
```

## 启动项目

1. 启动前后端服务：
```bash
cargo start
```

2. 访问前端界面：
打开浏览器访问 `http://localhost:5173`

## API 接口

- `POST /rembg/image` - 图像抠图
- `POST /rembg/mask` - 生成掩码

## 项目结构

```
.
├── backend/          # Rust 后端代码
├── frontend/         # React 前端代码
├── examples/         # 示例图片
└── .cargo/          # Cargo 配置
```

## 许可证

MIT 