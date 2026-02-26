// Theme Toggle Functionality
(function() {
    'use strict';

    // Check for saved theme preference or default to light mode
    const currentTheme = localStorage.getItem('theme') || 'light';

    // Function to set theme
    function setTheme(theme) {
        if (theme === 'dark') {
            document.documentElement.classList.add('dark-mode');
            localStorage.setItem('theme', 'dark');
            updateToggleButton('🌙');
        } else {
            document.documentElement.classList.remove('dark-mode');
            localStorage.setItem('theme', 'light');
            updateToggleButton('☀️');
        }
    }

    // Function to update toggle button icon
    function updateToggleButton(icon) {
        const toggleBtn = document.getElementById('theme-toggle');
        if (toggleBtn) {
            toggleBtn.textContent = icon;
            toggleBtn.setAttribute('aria-label',
                icon === '🌙' ? 'Switch to light mode' : 'Switch to dark mode');
        }
    }

    // Function to toggle theme
    function toggleTheme() {
        const isDark = document.documentElement.classList.contains('dark-mode');
        setTheme(isDark ? 'light' : 'dark');
    }

    // Apply saved theme on page load
    document.addEventListener('DOMContentLoaded', function() {
        setTheme(currentTheme);

        // Create toggle button if it doesn't exist
        if (!document.getElementById('theme-toggle')) {
            const toggleBtn = document.createElement('button');
            toggleBtn.id = 'theme-toggle';
            toggleBtn.setAttribute('aria-label', 'Toggle dark mode');
            toggleBtn.onclick = toggleTheme;
            document.body.appendChild(toggleBtn);

            // Set initial icon
            updateToggleButton(currentTheme === 'dark' ? '🌙' : '☀️');
        }
    });

    // Also apply immediately (before DOM ready) to prevent flash
    if (currentTheme === 'dark') {
        document.documentElement.classList.add('dark-mode');
    }
})();