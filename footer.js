class SiteFooter extends HTMLElement {
	connectedCallback() {
		this.innerHTML = `
      <footer class="border-t border-slate-900 bg-slate-950/80 py-8 px-4 w-full mt-auto">
        <div class="max-w-4xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <div>© Homelab Infrastructure</div>
          <div class="flex items-center gap-4">
            <!-- Email -->
            <a href="mailto:valentin@barbaza.org" class="flex items-center gap-1.5 hover:text-slate-300 transition-colors">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"/>
              </svg>
              E-Mail
            </a>
            <!-- Matrix -->
            <a href="https://matrix.to/#/@valentin:barbaza.org" target="_blank" class="flex items-center gap-1.5 hover:text-slate-300 transition-colors">
              <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="currentColor">
                <path d="M0.632 0.55v22.9h2.28v-1.14H1.77V1.69h1.142V0.55zm22.736 0v1.14h-1.14v20.62h1.14v1.14h-2.28V0.55zM8.135 7.158c-1.04 0-1.928.423-2.455 1.145V7.4h-1.63v9.442h1.63v-5.26c0-1.127.876-2.044 1.956-2.044 1.08 0 1.956.917 1.956 2.044v5.26h1.63v-5.26c0-1.127.876-2.044 1.956-2.044 1.08 0 1.956.917 1.956 2.044v5.26h1.63V11.23c0-2.316-1.878-4.072-4.14-4.072-1.282 0-2.392.56-3.083 1.488-.69-1-1.8-1.488-3.083-1.488z"/>
              </svg>
              Matrix
            </a>
          </div>
        </div>
      </footer>
    `;
	}
}

customElements.define('site-footer', SiteFooter);
