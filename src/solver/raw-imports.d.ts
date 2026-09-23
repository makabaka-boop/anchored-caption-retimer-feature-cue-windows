// Vitest/Vite raw imports used by tests (e.g. asserting on source text).
declare module '*?raw' {
  const content: string;
  export default content;
}
