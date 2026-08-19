# 私厨 AI（图片识别生成菜谱）项目命令管理
# 用法：make <target>  或  make help 查看全部命令
#
# 项目结构（设计报告 v3.0）：
#   app/            后端（FastAPI + LangGraph + Checkpointer）
#   web/            前端（Next.js，静态导出到 app/static/）
#   chief.py        主脑 Agent 参考实现（Agnes 多模态 + Tavily）

SHELL := /bin/bash
PY := .venv/bin/python
PORT ?= 8000

.PHONY: help install install-backend install-frontend backend backend-stop backend-log web-dev web-build web-deploy web-sync chief stop check clean

help: ## 查看所有可用命令
	@echo "私厨 AI 项目命令管理："
	@echo ""
	@awk -F'## ' '/^[a-zA-Z_-]+:.*## / {printf "  %-18s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "说明：PORT 默认 8000，可用 make backend PORT=9000 覆盖"

## ---------- 安装 ----------

install: install-backend install-frontend ## 安装后端 + 前端全部依赖

install-backend: ## 安装后端依赖（.venv + requirements.txt）
	.venv/bin/pip install -r requirements.txt

install-frontend: ## 安装前端依赖（web/ npm install）
	cd web && npm install --no-fund --no-audit

## ---------- 后端 ----------

backend: ## 启动后端服务（--reload 热重载，默认端口 8000）
	$(PY) -m uvicorn app.main:app --host 0.0.0.0 --port $(PORT) --reload

backend-stop: ## 停止后端服务
	-pkill -f "uvicorn app.main:app"
	@echo "后端已停止（若未在运行则忽略）"

backend-log: ## 查看后端运行日志（配合 nohup 后台运行使用）
	@tail -f /tmp/sichu.log

## ---------- 前端（Next.js） ----------

web-dev: ## 启动前端开发服务器（浏览器调试，默认 http://localhost:3000）
	cd web && npm run dev

web-build: ## 构建前端静态产物（web/out）
	cd web && npm run build

web-deploy: ## 构建并部署到 app/static/（FastAPI 静态托管）
	cd web && npm run build && rm -rf ../app/static/* && cp -r out/. ../app/static/

web-sync: ## 一键构建前端并同步到 app/static/（支持 web-sync ARGS=--skip-build 跳过构建）
	bash scripts/web-sync.sh $(ARGS)

stop: backend-stop ## 一键停止后端
	@echo "后端已停止（前端开发服务器用 Ctrl+C 停止）"

## ---------- 演示脚本 ----------

chief: ## 运行主脑 Agent（Agnes 多模态 + Tavily 搜索，支持传问题参数）
	$(PY) chief.py $(ARGS)

## ---------- 检查 / 清理 ----------

check: ## 验证后端核心包可导入
	$(PY) -c "import fastapi, uvicorn, langchain; from importlib.metadata import version; print('fastapi', fastapi.__version__); print('langchain', langchain.__version__); print('langgraph', version('langgraph'))"

clean: ## 清理运行产物（SQLite 数据、前端构建产物、__pycache__）
	rm -rf data/*.db web/out web/.next
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "清理完成"
