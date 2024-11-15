const CACHE_NAME = 'v1'; // Increment this to invalidate old caches
const ASSETS_TO_CACHE = [
  '/', // Cache the root URL
  // '/index.html',
  // '/styles.css',
  // '/script.js',
  // '/images/logo.png',
  // Add other assets like images, fonts, etc.
];

// Install event - caches all necessary assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('Caching files');
      return cache.addAll(ASSETS_TO_CACHE);
    })
  );
});

// Activate event - cleans up old caches if necessary
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) =>
      Promise.all(
        cacheNames.map((cache) => {
          if (cache !== CACHE_NAME) {
            console.log('Deleting old cache:', cache);
            return caches.delete(cache);
          }
        })
      )
    )
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.method === 'POST') {
    return; // Ignore POST requests for caching.
  }
  event.respondWith(
    fetch(event.request)
      .then((networkResponse) => {
        return caches.open(CACHE_NAME).then((cache) => {
          console.log('fetched from network, now caching')
          cache.put(event.request, networkResponse.clone());
          return networkResponse;
        });
      })
      .catch(() => caches.match(event.request))
  );
});
