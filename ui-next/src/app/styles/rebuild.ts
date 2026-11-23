// The rebuild's own stylesheets (styles/rebuild/*.css), in name order: imported by index.ts after the concept's.
import.meta.glob('./rebuild/*.css', { eager: true });
