// --- DETECT & INITIALIZE SYSTEM / USER THEME ---
function applyTheme() {
	const userChoice = localStorage.getItem('theme');
	const systemPrefersDark = globalThis.matchMedia('(prefers-color-scheme: dark)').matches;

	// Use saved choice if exists, otherwise fallback to system setting
	document.documentElement.classList.toggle('dark', userChoice === 'dark' || (!userChoice && systemPrefersDark));

	updateThemeIcons();
}

// Run immediately to avoid page flicker
applyTheme();

// Listen for device/OS theme changes in real-time (e.g., sunset auto-switch)
globalThis.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
	// Only adapt if the user hasn't explicitly clicked the manual toggle
	if (!localStorage.getItem('theme')) {
		applyTheme();
	}
});

globalThis.toggleTheme = function () {
	const isDark = document.documentElement.classList.toggle('dark');
	// Save explicit preference
	localStorage.setItem('theme', isDark ? 'dark' : 'light');
	updateThemeIcons();
};

function updateThemeIcons() {
	const isDark = document.documentElement.classList.contains('dark');
	const sun = document.querySelector('#sun-icon');
	const moon = document.querySelector('#moon-icon');
	if (sun && moon) {
		sun.classList.toggle('hidden', !isDark);
		moon.classList.toggle('hidden', isDark);
	}
}

// --- CONFIGURE YOUR TABS HERE ---
const NAV_ITEMS = [
	{name: 'Home', href: 'index.html'},
	{name: 'Services', href: 'services.html'},
	{name: 'Status', href: 'status.html'},
	// To add a new page later, just add: { name: 'About', href: 'about.html' }
];

// --- HEADER COMPONENT ---
class SiteHeader extends HTMLElement {
	connectedCallback() {
		// Detect which HTML file is currently open
		let currentPath = globalThis.location.pathname.split('/').pop();
		// Default to index.html if the URL ends in '/'
		if (!currentPath || currentPath === '') {currentPath = 'index.html';}

		// Generate tabs dynamically
		const navLinks = NAV_ITEMS.map(item => {
			const isActive = currentPath === item.href;
			return `
        <a href="${item.href}" class="px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
			isActive
				? 'bg-slate-800 text-white shadow-sm'
				: 'text-slate-400 hover:text-white'
		}">
          ${item.name}
        </a>
      `;
		}).join('');

   const isDark = document.documentElement.classList.contains('dark');

    this.innerHTML = `
        <header class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-10 pb-6 border-b border-slate-200 dark:border-slate-800/80">
          <div>
            <h1 class="text-2xl font-bold tracking-tight text-slate-900 dark:text-white flex items-center gap-2">
              <span class="w-3 h-3 rounded-full bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.6)]"></span>
              Homelab Portal
            </h1>
            <p class="text-sm text-slate-500 dark:text-slate-400 mt-1">Self-hosted infrastructure & services</p>
          </div>

          <div class="flex items-center gap-2">
            <!-- Navigation Tabs -->
            <nav class="flex bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 p-1 rounded-xl">
              ${navLinks}
            </nav>

            <!-- Theme Toggle Button -->
            <button onclick="toggleTheme()" aria-label="Toggle Theme" class="p-2 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-slate-900 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-all shadow-sm">
              <!-- Sun Icon (Shows in Dark Mode) -->
              <svg id="sun-icon" class="w-4 h-4 ${isDark ? '' : 'hidden'}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 9h-1m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
              </svg>
              <!-- Moon Icon (Shows in Light Mode) -->
              <svg id="moon-icon" class="w-4 h-4 ${isDark ? 'hidden' : ''}" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
              </svg>
            </button>
          </div>
        </header>
      `;
    }
  }
}

// --- 3. FOOTER COMPONENT ---
class SiteFooter extends HTMLElement {
	connectedCallback() {
		this.innerHTML = `
      <footer class="border-t border-slate-900 bg-slate-950/80 py-8 px-4 w-full mt-auto">
        <div class="max-w-4xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <div>Barbaza's website</div>
          <div class="flex items-center gap-4">
            <!-- Email -->
            <a href="mailto:valentin@barbaza.org" target="_blank" class="flex items-center gap-1.5 hover:text-slate-300 transition-colors">
              <img class="w-4 h-4 object-contain shrink-0 brightness-0 invert opacity-70 hover:opacity-100 transition-opacity logo" src="logos/email.svg" alt="E-Mail Logo">
              E-Mail
            </a>
            <!-- Matrix -->
            <a href="https://matrix.to/#/@valentin:barbaza.org" target="_blank" class="flex items-center gap-1.5 hover:text-slate-300 transition-colors">
              <img class="w-4 h-4 object-contain shrink-0 brightness-0 invert opacity-70 hover:opacity-100 transition-opacity logo" src="logos/matrix.svg" alt="The Matrix Logo">
              Matrix
            </a>
          </div>
        </div>
      </footer>
    `;
	}
}

// Register both tags
customElements.define('site-header', SiteHeader);
customElements.define('site-footer', SiteFooter);
