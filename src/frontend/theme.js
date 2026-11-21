// Light / dark theme: follows the system by default, a toggle button overrides it (saved in localStorage).
(function () {
    const KEY = "theme";
    const root = document.documentElement;
    const systemQuery = window.matchMedia("(prefers-color-scheme: dark)");

    function getStored() {
        try { return localStorage.getItem(KEY); } catch (e) { return null; }
    }
    function setStored(value) {
        try { localStorage.setItem(KEY, value); } catch (e) { /* storage blocked: the choice just won't persist */ }
    }
    function currentTheme() {
        return getStored() || (systemQuery.matches ? "dark" : "light");
    }

    // Chart.js reads its colors from Chart.defaults, so we copy our CSS variables into them
    function applyChartTheme() {
        if (typeof Chart === "undefined") return;
        const styles = getComputedStyle(root);
        Chart.defaults.font.family = styles.getPropertyValue("--font").trim();
        Chart.defaults.color = styles.getPropertyValue("--chart-text").trim();
        Chart.defaults.borderColor = styles.getPropertyValue("--chart-grid").trim();
        // Redraw charts that already exist (update() re-reads the defaults)
        Object.values(Chart.instances).forEach(chart => chart.update());
    }

    function updateButton() {
        const button = document.getElementById("theme-toggle");
        if (!button) return;
        const isDark = root.getAttribute("data-theme") === "dark";
        button.textContent = isDark ? "☀️" : "🌙";
        button.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
        button.title = button.getAttribute("aria-label");
    }

    function applyTheme(theme) {
        root.setAttribute("data-theme", theme);
        updateButton();
        applyChartTheme();
    }

    // Runs immediately (script is in <head>) so the page never flashes the wrong theme
    applyTheme(currentTheme());

    document.addEventListener("DOMContentLoaded", () => {
        const links = document.querySelector(".nav-links");
        if (links) {
            const button = document.createElement("button");
            button.id = "theme-toggle";
            button.className = "theme-toggle";
            button.type = "button";
            button.addEventListener("click", () => {
                const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
                setStored(next);
                applyTheme(next);
            });
            links.appendChild(button);
            updateButton();
        }
    });

    // If the person never chose manually, keep following the system setting
    systemQuery.addEventListener("change", () => {
        if (!getStored()) applyTheme(currentTheme());
    });
})();