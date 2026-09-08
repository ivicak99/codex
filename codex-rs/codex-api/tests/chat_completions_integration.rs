use anyhow::Context;
use anyhow::Result;
use codex_api::AuthProvider;
use codex_api::ChatCompletionsClient;
use codex_api::Provider;
use codex_api::Reasoning;
use codex_api::ReasoningContext;
use codex_api::ResponsesApiRequest;
use codex_api::RetryConfig;
use codex_client::ReqwestTransport;
use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::openai_models::ReasoningEffort;
use futures::StreamExt;
use http::HeaderMap;
use pretty_assertions::assert_eq;
use serde_json::Value;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;
use wiremock::Mock;
use wiremock::MockServer;
use wiremock::ResponseTemplate;
use wiremock::matchers::method;
use wiremock::matchers::path;

#[derive(Clone)]
struct NoAuth;

impl AuthProvider for NoAuth {
    fn add_auth_headers(&self, _headers: &mut HeaderMap) {}
}

async fn captured_request(host: &str, reasoning: Option<Reasoning>) -> Result<Value> {
    let server = MockServer::start().await;
    let port = server.address().port();
    // Explicit DNS resolution keeps every request on this test's loopback
    // server, including the production hostname used to choose the dialect.
    let http_client = reqwest::Client::builder()
        .no_proxy()
        .resolve(host, *server.address())
        .build()
        .context("local test client")?;
    let provider = Provider {
        name: "test".to_string(),
        base_url: format!("http://{host}:{port}/openrouter.ai/v1"),
        query_params: None,
        headers: HeaderMap::new(),
        retry: RetryConfig {
            max_attempts: 1,
            base_delay: Duration::from_millis(1),
            retry_429: false,
            retry_5xx: false,
            retry_transport: false,
        },
        stream_idle_timeout: Duration::from_secs(1),
    };
    Mock::given(method("POST"))
        .and(path("/openrouter.ai/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string("data: [DONE]\n\n"),
        )
        .expect(1)
        .mount(&server)
        .await;
    let client = ChatCompletionsClient::new(
        ReqwestTransport::new(http_client),
        provider,
        Arc::new(NoAuth),
    );
    let request = ResponsesApiRequest {
        model: "provider-model".to_string(),
        instructions: "A short instruction".to_string(),
        input: vec![],
        tools: None,
        tool_choice: "auto".to_string(),
        parallel_tool_calls: false,
        reasoning,
        store: false,
        stream: true,
        stream_options: None,
        include: vec![],
        service_tier: None,
        prompt_cache_key: None,
        text: None,
        client_metadata: None,
    };
    let mut stream = client
        .stream_request(request)
        .await
        .context("chat request")?;
    while let Some(event) = stream.next().await {
        event.context("chat stream event")?;
    }
    let requests = server
        .received_requests()
        .await
        .context("captured requests")?;
    assert_eq!(requests.len(), 1);
    serde_json::from_slice(&requests[0].body).context("JSON request body")
}

fn expected_request() -> Value {
    json!({
        "model": "provider-model",
        "messages": [{"role": "system", "content": "A short instruction"}],
        "tool_choice": "auto",
        "stream": true
    })
}

#[tokio::test]
async fn explicit_reasoning_effort_reaches_chat_completions_body() -> Result<()> {
    for effort in [
        ReasoningEffort::High,
        ReasoningEffort::None,
        ReasoningEffort::Custom("adaptive".to_string()),
    ] {
        let mut expected = expected_request();
        expected["reasoning_effort"] = json!(effort);
        let reasoning = Reasoning {
            effort: Some(effort),
            summary: Some(ReasoningSummary::Detailed),
            context: Some(ReasoningContext::AllTurns),
        };
        let actual = captured_request("api.example.test", Some(reasoning)).await?;
        assert_eq!(actual, expected);
    }
    Ok(())
}

#[tokio::test]
async fn provider_default_omits_reasoning_fields_in_both_dialects() -> Result<()> {
    for host in ["api.example.test", "openrouter.ai"] {
        for reasoning in [
            None,
            Some(Reasoning {
                effort: None,
                summary: Some(ReasoningSummary::Detailed),
                context: Some(ReasoningContext::AllTurns),
            }),
        ] {
            let actual = captured_request(host, reasoning).await?;
            assert_eq!(actual, expected_request());
        }
    }
    Ok(())
}

#[tokio::test]
async fn openrouter_receives_only_nested_effort() -> Result<()> {
    let actual = captured_request(
        "openrouter.ai",
        Some(Reasoning {
            effort: Some(ReasoningEffort::Max),
            summary: Some(ReasoningSummary::Detailed),
            context: Some(ReasoningContext::AllTurns),
        }),
    )
    .await?;
    let mut expected = expected_request();
    expected["reasoning"] = json!({"effort": "max"});
    assert_eq!(actual, expected);
    Ok(())
}

#[tokio::test]
async fn openrouter_name_in_another_host_or_path_does_not_change_dialect() -> Result<()> {
    let actual = captured_request(
        "openrouter.ai.example.test",
        Some(Reasoning {
            effort: Some(ReasoningEffort::High),
            summary: None,
            context: None,
        }),
    )
    .await?;
    let mut expected = expected_request();
    expected["reasoning_effort"] = json!("high");
    assert_eq!(actual, expected);
    Ok(())
}
