// A file handed to the browser to save (a revision exported as JSON, the copies of what storage could not read): a
// plain download, as the concept does when it is opened as a local file.

export function saveFile(name: string, text: string): void {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 3000);
}
