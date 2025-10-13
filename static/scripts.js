document.getElementById("submitBtn").addEventListener("click", getRecommendations);

async function getRecommendations() {
  const userId = document.getElementById("user_id").value.trim();
  const resultsDiv = document.getElementById("results");
  const spinner = document.getElementById("spinner");

  resultsDiv.innerHTML = "";
  spinner.classList.remove("hidden");

  if (!userId) {
    spinner.classList.add("hidden");
    resultsDiv.innerHTML = `<p style="color:red;">⚠️ Please enter a valid User ID.</p>`;
    return;
  }

  try {
    const response = await fetch("/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_id: userId })
    });

    const data = await response.json();
    spinner.classList.add("hidden");

    if (data.error) {
      resultsDiv.innerHTML = `<p style="color:red;">❌ ${data.error}</p>`;
      return;
    }

    const recs = data.recommendations || [];
    if (recs.length === 0) {
      resultsDiv.innerHTML = "<p>No recommendations found.</p>";
      return;
    }

    const html = [`<h2>Recommendations for ${data.user_id}</h2>`];
    for (const rec of recs) {
      html.push(`
        <div class="recommendation">
          <div class="product-title">🛍️ ${rec.product_name} (${rec.product_id})</div>
          <div class="meta">⭐ Score: ${rec.score.toFixed(4)} | Source: ${rec.source_event}</div>
          <div class="explanation">💬 ${rec.explanation}</div>
          <div class="evidence">🔗 Evidence: ${rec.evidence?.length ? rec.evidence.join(", ") : "None"}</div>
        </div>
      `);
    }

    resultsDiv.innerHTML = html.join("");
  } catch (err) {
    spinner.classList.add("hidden");
    resultsDiv.innerHTML = `<p style="color:red;">❌ Failed to fetch recommendations: ${err}</p>`;
  }
}

