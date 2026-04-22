const CACHE_NAME = 'smartdoc-cache-v1';
const STATIC_ASSETS = [
  '/',
  '/static/style.css',
  '/static/main.js',
  '/static/manifest.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // DO NOT cache API calls or the live video feed
  if (url.pathname.startsWith('/video_feed') || url.pathname.startsWith('/capture') || url.pathname.startsWith('/set_source') || url.pathname.startsWith('/rotate')) {
    return; 
  }

  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request);
    })
  );
});