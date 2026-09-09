// --- 1. CONFIGURE YOUR TABS HERE ---
const NAV_ITEMS = [
	{name: 'Home', href: 'index.html'},
	{name: 'Status', href: 'status.html'},
	// {name: 'Services', href: 'services.html'},
	// To add a new page later, just add: { name: 'About', href: 'about.html' }
];

// --- 2. HEADER COMPONENT ---
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

		this.innerHTML = `
      <header class="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-10 pb-6 border-b border-slate-800/80">
        <div>
          <h1 class="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <span class="w-3 h-3 rounded-full bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.6)]"></span>
            Homelab Portal
          </h1>
          <p class="text-sm text-slate-400 mt-1">Self-hosted infrastructure & services</p>
        </div>

        <nav class="flex bg-slate-900 border border-slate-800 p-1 rounded-xl">
          ${navLinks}
        </nav>
      </header>
    `;
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
