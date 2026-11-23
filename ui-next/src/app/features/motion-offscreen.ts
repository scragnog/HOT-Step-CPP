// The decorative loops of a tile scrolled out of view pause (18-motion.js: .offscreen, 35-motion.css): each tile on
// the desk is watched as it arrives.

const io = new IntersectionObserver(es => es.forEach(x => x.target.classList.toggle('offscreen', !x.isIntersecting)), { rootMargin: '120px' });
const seen = new WeakSet<Element>();
const watch = () => document.querySelectorAll('#bento > .tile').forEach(t => { if (!seen.has(t)) { seen.add(t); io.observe(t); } });
const start = () => { const g = document.getElementById('bento'); if (!g) { requestAnimationFrame(start); return; } watch(); new MutationObserver(watch).observe(g, { childList: true }); };
requestAnimationFrame(start);
