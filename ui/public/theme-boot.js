// Prevent a flash of the wrong theme: apply the dark class before any CSS loads.
// A file, not an inline script, so the CSP can keep script-src 'self'.
(function () {
  // Skin (docs/dev/skins.md): same early apply, so there is no flash. main.tsx drops it if it does not target the shell.
  try {
    var skin = localStorage.getItem('hs-skin');
    if (skin && /^[a-z][a-z0-9-]{1,31}$/.test(skin)) document.documentElement.setAttribute('data-skin', skin);
  } catch (e) { /* storage blocked: no skin */ }
  try {
    if (localStorage.getItem('hs-theme') === 'light') return; // stay light
    document.documentElement.classList.add('dark'); // default to dark
  } catch (e) { document.documentElement.classList.add('dark'); }
})();
