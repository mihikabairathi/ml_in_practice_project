// Seed recipes for the Cold-Start Onboarding Quiz
const SEED_RECIPES = [
  { id: 38, title: "Low Carb Creamy Garlic Chicken", minutes: 25, ingredients: 6 },
  { id: 244, title: "Classic Italian Tomato & Basil Pasta", minutes: 20, ingredients: 5 },
  { id: 521, title: "Slow-Cooked Beef Stew with Root Vegetables", minutes: 120, ingredients: 10 },
  { id: 890, title: "Fresh Lemon Herb Grilled Salmon Salad", minutes: 15, ingredients: 7 }
];

// Persistent state for cold-start user session
let userSeedRatings = {};
let userId = "user_" + Math.random().toString(36).substring(2, 9);

document.addEventListener("DOMContentLoaded", () => {
  renderSeedQuiz();
  setupEventListeners();
});

function renderSeedQuiz() {
  const container = document.getElementById("seed-recipes-container");
  container.innerHTML = SEED_RECIPES.map(recipe => `
    <div class="rec-card">
      <div>
        <h3>${recipe.title}</h3>
        <p class="rec-meta">⏱️ ${recipe.minutes} mins | 🥗 ${recipe.ingredients} ingredients</p>
      </div>
      <div class="rating-actions">
        <label>Rate: </label>
        <select onchange="updateSeedRating(${recipe.id}, this.value)">
          <option value="0">Not rated</option>
          <option value="5">⭐⭐⭐⭐⭐ Love it</option>
          <option value="4">⭐⭐⭐⭐ Like it</option>
          <option value="3">⭐⭐⭐ Neutral</option>
          <option value="1">⭐ Dislike</option>
        </select>
      </div>
    </div>
  `).join("");
}

function updateSeedRating(recipeId, rating) {
  if (parseInt(rating) > 0) {
    userSeedRatings[recipeId] = parseInt(rating);
  } else {
    delete userSeedRatings[recipeId];
  }
}

function setupEventListeners() {
  const form = document.getElementById("preference-form");
  const resetBtn = document.getElementById("reset-btn");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    await fetchLiveRecommendations();
  });

  resetBtn.addEventListener("click", () => {
    document.getElementById("quiz-section").style.display = "block";
    document.getElementById("results-section").style.display = "none";
  });
}

async function fetchLiveRecommendations() {
  const maxTime = parseInt(document.getElementById("max-time").value) || 60;
  const maxIngredients = parseInt(document.getElementById("max-ingredients").value) || 20;
  
  const quizSection = document.getElementById("quiz-section");
  const resultsSection = document.getElementById("results-section");
  const grid = document.getElementById("rec-grid");

  quizSection.style.display = "none";
  resultsSection.style.display = "block";
  grid.innerHTML = `<p class="loading">Running model inference on your ratings...</p>`;

  const payload = {
    user_id: userId,
    seed_ratings: userSeedRatings,
    max_time: maxTime,
    max_ingredients: maxIngredients
  };

  try {
    const response = await fetch("http://localhost:8000/api/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (data.status === "success" && data.recommendations.length) {
      grid.innerHTML = data.recommendations.map(item => `
        <div class="rec-card">
          <div>
            <h3>${escapeHtml(item.title)}</h3>
            <div class="rec-meta">
              <span>⏱️ ${item.minutes} mins</span> | 
              <span>🥗 ${item.n_ingredients} ingredients</span>
            </div>
          </div>
          <div class="rating-actions" style="margin-top: 1rem;">
            <button class="btn-sm" onclick="logUserFeedback(${item.recipe_id}, 'like')">👍 Like</button>
            <button class="btn-sm" onclick="logUserFeedback(${item.recipe_id}, 'dislike')">👎 Dislike</button>
          </div>
        </div>
      `).join("");
    } else {
      grid.innerHTML = "<p>No recipes matched your criteria.</p>";
    }
  } catch (err) {
    console.error("Inference request failed:", err);
    grid.innerHTML = `<p style="color: red;">Error connecting to Python inference backend at http://localhost:8000.</p>`;
  }
}

function logUserFeedback(recipeId, action) {
  console.log(`[Feedback logged] Session: ${userId} | Recipe: ${recipeId} | Action: ${action}`);
  alert(`Feedback recorded for recipe #${recipeId}!`);
}

function escapeHtml(str) {
  return str.replace(/[&<>'"]/g, 
    tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
  );
}