# 通义听悟 API 管理台

一个本地运行的 Python Web 项目，用于调用通义听悟 OpenAPI。

## 已支持

- 在配置中心保存 AccessKey ID、AccessKey Secret 与 AppKey（仅写入本机 `.env`）
- 创建离线音视频转写任务（支持上传本地文件自动生成 URL）
- 按 TaskId 查询任务状态与结果
- 热词词表的创建、列表、查看、更新和删除
- 独立「实时转写」页面：创建会议 → 麦克风录音推流 → 实时转写显示，支持暂停/恢复/结束

## 使用

1. 创建虚拟环境并安装依赖：`pip install -r requirements.txt`
2. 启动服务：`python app.py`
3. 在浏览器打开 `http://127.0.0.1:5000`，先在“配置中心”填写凭证。

离线转写的文件地址必须是听悟服务可访问的 HTTP/HTTPS 音视频 URL。AccessKey 建议使用最小权限 RAM 用户，不要使用主账号密钥。

