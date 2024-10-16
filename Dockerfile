# 使用 Python 3.12 作为基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 添加应用程序源代码到工作目录
ADD . /app

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 设置默认命令
CMD ["python", "app.py"]
