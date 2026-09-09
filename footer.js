class SiteFooter extends HTMLElement {
	connectedCallback() {
		this.innerHTML = `
      <footer class="border-t border-slate-900 bg-slate-950/80 py-8 px-4 w-full mt-auto">
        <div class="max-w-4xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-500">
          <div>© Homelab Infrastructure</div>
          <div class="flex items-center gap-4">
            <!-- Email -->
<img class="logo" src="logos/email.svg" alt="E-Mail Logo">
              E-Mail
            </a>
            <!-- Matrix -->
            <a href="https://matrix.to/#/@valentin:barbaza.org" target="_blank" class="flex items-center gap-1.5 hover:text-slate-300 transition-colors">
<img class="logo" src="logos/matrix.svg" alt="The Matrix Logo">
              Matrix
            </a>
          </div>
        </div>
      </footer>
    `;
	}
}

customElements.define('site-footer', SiteFooter);
