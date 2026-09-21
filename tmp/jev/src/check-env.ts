const PLACEHOLDER = "your_ai_gateway_api_key";
const MODELS_URL = "https://ai-gateway.vercel.sh/v1/models";

const key = process.env.AI_GATEWAY_API_KEY?.trim() ?? "";

if (!key || key === PLACEHOLDER) {
  console.error(
    "AI_GATEWAY_API_KEY 未配置。请到 Vercel Dashboard 创建 AI Gateway key，写入 tmp/jev/.env.local。",
  );
  console.error("文档: https://vercel.com/docs/ai-gateway/authentication-and-byok/api-keys");
  process.exit(1);
}

const response = await fetch(MODELS_URL, {
  headers: {
    Authorization: `Bearer ${key}`,
  },
});

if (!response.ok) {
  const body = await response.text();
  console.error(`网关鉴权失败: HTTP ${response.status}`);
  console.error(body.slice(0, 500));
  process.exit(1);
}

const payload = (await response.json()) as {
  data?: Array<{ id?: string }>;
};

const models = (payload.data ?? []).map((item) => item.id).filter(Boolean);
console.log(`环境就绪: Node ${process.version}, 密钥已加载, 可访问模型 ${models.length} 个`);
if (models.length > 0) {
  console.log(`示例模型: ${models.slice(0, 8).join(", ")}`);
}
