((global) => {
  "use strict"

  function normalizePath(value) {
    const path = new URL(value || "/", "https://amoursec.github.io").pathname || "/"
    return path === "/" || path.endsWith("/") ? path : `${path}/`
  }

  function formatCount(value) {
    return new Intl.NumberFormat("zh-CN").format(value)
  }

  async function render() {
    if (document.querySelector("[data-pageviews]")) return
    const heading = document.querySelector(".md-content__inner > h1")
    if (!heading) return

    try {
      const script = document.querySelector('script[src$="/assets/javascripts/pageviews.js"]')
      const scriptUrl = script?.src ? new URL(script.src) : null
      const snapshotUrl = scriptUrl
        ? new URL("../data/pageviews.json", scriptUrl).href
        : "/assets/data/pageviews.json"
      const response = await fetch(snapshotUrl, { cache: "no-store" })
      if (!response.ok) return
      const snapshot = await response.json()
      const siteRoot = scriptUrl ? new URL("../../", scriptUrl).pathname : "/"
      const pagePath = global.location.pathname.startsWith(siteRoot)
        ? `/${global.location.pathname.slice(siteRoot.length)}`
        : global.location.pathname
      const count = snapshot.pages?.[normalizePath(pagePath)]
      if (!Number.isInteger(count) || count <= 0) return

      const counter = document.createElement("p")
      counter.className = "page-views"
      counter.dataset.pageviews = ""
      counter.textContent = `浏览量：${formatCount(count)}`
      heading.insertAdjacentElement("afterend", counter)
    } catch {
      return
    }
  }

  global.AIKGPageviews = { normalizePath, formatCount, render }

  if (typeof document !== "undefined") {
    if (typeof document$ !== "undefined") {
      document$.subscribe(render)
    } else if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", render, { once: true })
    } else {
      void render()
    }
  }
})(globalThis)
