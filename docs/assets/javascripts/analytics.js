(() => {
  "use strict"

  if (location.hostname !== "amoursec.github.io") return

  const tracker = document.createElement("script")
  tracker.async = true
  tracker.src = "https://gc.zgo.at/count.js"
  tracker.dataset.goatcounter = "https://amoursec.goatcounter.com/count"
  document.head.appendChild(tracker)
})()
