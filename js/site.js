/* Shared navigation, theme, progress bar and voice narration for every page. */
(function () {
  const PAGES = [
    { slug: "index", title: "Home", group: "Start" },
    { slug: "01-intro", n: 1, title: "Temporal Instability in Pose Estimation", group: "Tutorial" },
    { slug: "02-trajectories", n: 2, title: "Pose Landmark Trajectories", group: "Tutorial" },
    { slug: "03-simple-smoothing", n: 3, title: "Simple Smoothing Methods", group: "Tutorial" },
    { slug: "04-one-euro", n: 4, title: "The One Euro Filter", group: "Tutorial" },
    { slug: "05-kalman", n: 5, title: "Kalman Filtering", group: "Tutorial" },
    { slug: "06-missing-keypoints", n: 6, title: "Missing & Low-Confidence Keypoints", group: "Tutorial" },
    { slug: "07-smoothness-latency", n: 7, title: "Smoothness vs. Latency", group: "Tutorial" },
    { slug: "08-exercise-analysis", n: 8, title: "Applications to Exercise Analysis", group: "Tutorial" },
    { slug: "09-experiment", n: 9, title: "Experimental Comparison", group: "Tutorial" },
    { slug: "10-limitations", n: 10, title: "Limitations & Future Work", group: "Tutorial" },
    { slug: "lab", title: "Interactive Filter Lab", group: "Try it", icon: "▶" },
    { slug: "webcam", title: "Live Webcam Lab", group: "Try it", icon: "◉" },
    { slug: "quiz", title: "Self-Check Quiz", group: "Try it", icon: "?" },
    { slug: "bibliography", title: "Annotated Bibliography", group: "Reference", icon: "§" },
  ];

  const body = document.body;
  const current = body.dataset.page || "index";
  const idx = PAGES.findIndex(p => p.slug === current);
  const href = p => (p.slug === "index" ? "index.html" : p.slug + ".html");

  // ---- theme (remember choice per viewer; safe if storage is blocked)
  const root = document.documentElement;
  try { const t = localStorage.getItem("theme"); if (t) root.dataset.theme = t; } catch (e) {}
  function toggleTheme() {
    const dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch (e) {}
    window.dispatchEvent(new Event("themechange"));
  }

  // ---- top bar
  const top = document.createElement("header");
  top.className = "topbar";
  top.innerHTML = `
    <button class="iconbtn" id="menuBtn" aria-label="Open navigation">☰ Menu</button>
    <a class="brand" href="index.html">Stabilizing Pose Trajectories<small>CS663 Research Tutorial</small></a>
    <span class="spacer"></span>
    <button class="iconbtn" id="themeBtn" aria-label="Toggle dark mode">◐ Theme</button>
    <div class="progress" id="progress"></div>`;

  // ---- sidebar
  const side = document.createElement("nav");
  side.className = "sidebar";
  side.setAttribute("aria-label", "Tutorial sections");
  let lastGroup = "";
  let html = "";
  PAGES.forEach(p => {
    if (p.group !== lastGroup) { html += `<div class="group">${p.group}</div>`; lastGroup = p.group; }
    const mark = p.n ? p.n : (p.icon || "⌂");
    html += `<a href="${href(p)}" class="${p.slug === current ? "active" : ""}"><span class="n">${mark}</span><span>${p.title}</span></a>`;
  });
  html += `<div class="group">Time</div><div class="meta" style="padding:0 10px">About 20–30 minutes in total. Each section takes 2–3 minutes; the labs take 5.</div>`;
  side.innerHTML = html;

  // ---- wrap existing <main>
  const main = document.querySelector("main");
  const layout = document.createElement("div");
  layout.className = "layout";
  main.parentNode.insertBefore(layout, main);
  layout.appendChild(side);
  layout.appendChild(main);
  body.insertBefore(top, layout);

  document.getElementById("themeBtn").onclick = toggleTheme;
  document.getElementById("menuBtn").onclick = () => body.classList.toggle("nav-open");
  main.addEventListener("click", () => body.classList.remove("nav-open"));

  // ---- reading progress
  const bar = document.getElementById("progress");
  const onScroll = () => {
    const h = document.documentElement.scrollHeight - innerHeight;
    bar.style.width = (h > 0 ? (scrollY / h) * 100 : 0) + "%";
  };
  addEventListener("scroll", onScroll, { passive: true }); onScroll();

  // ---- prev / next
  const content = main.querySelector(".content") || main;
  if (idx >= 0) {
    const prev = PAGES[idx - 1], next = PAGES[idx + 1];
    const pager = document.createElement("nav");
    pager.className = "pager";
    pager.innerHTML =
      (prev ? `<a class="prev" href="${href(prev)}"><small>← Previous</small>${prev.title}</a>` : `<span class="ph"></span>`) +
      (next ? `<a class="next" href="${href(next)}"><small>Next →</small>${next.title}</a>` : `<span class="ph"></span>`);
    content.appendChild(pager);
  }

  // ---- voice narration
  // Each page has <div id="voice"></div>. If audio/<slug>.mp3 exists it plays the
  // recorded narration; otherwise a "Read aloud" button uses the browser's voice
  // to read the page's narration script (<script type="text/plain" id="narration">).
  const vbox = document.getElementById("voice");
  if (vbox) {
    vbox.className = "voice";
    const script = document.getElementById("narration");
    const text = script ? script.textContent.trim().replace(/\s+/g, " ") : "";
    const src = `audio/${current}.mp3`;
    vbox.innerHTML = `<span class="label">🔊 Narration</span>
      <audio controls preload="none" src="${src}"></audio>
      <span class="hint">Recorded narration by the author.</span>`;
    const audio = vbox.querySelector("audio");
    const fallback = () => {
      if (!("speechSynthesis" in window) || !text) {
        vbox.innerHTML = `<span class="label">🔊 Narration</span><span class="hint">Narration audio is not available in this browser.</span>`;
        return;
      }
      vbox.innerHTML = `<span class="label">🔊 Narration</span>
        <button class="btn" id="ttsPlay">▶ Listen to this page</button>
        <button class="btn secondary" id="ttsStop">■ Stop</button>
        <span class="hint">Plays this page's narration script using your browser's voice.</span>`;
      document.getElementById("ttsPlay").onclick = () => {
        speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        u.rate = 1.0; speechSynthesis.speak(u);
      };
      document.getElementById("ttsStop").onclick = () => speechSynthesis.cancel();
    };
    audio.addEventListener("error", fallback);
    // probe whether the file exists without downloading it all
    fetch(src, { method: "HEAD" }).then(r => { if (!r.ok) fallback(); }).catch(fallback);
  }
})();
