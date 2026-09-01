(() => {
  "use strict"

  const tracker = document.createElement("script")
  tracker.defer = true
  tracker.src = "https://cloud.umami.is/script.js"
  tracker.dataset.websiteId = "e6bcb0cd-aee7-4383-8557-9cf7564c86a0"
  tracker.dataset.domains = "amoursec.github.io"
  tracker.dataset.excludeSearch = "true"
  tracker.dataset.excludeHash = "true"
  document.head.appendChild(tracker)
})()
