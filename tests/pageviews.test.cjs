"use strict"

const assert = require("node:assert/strict")

async function main() {
  require("../docs/assets/javascripts/pageviews.js")

  // Given clean and decorated article URLs
  const paths = ["", "/docs", "/docs/?q=1#part"]

  // When browser helpers normalize and format their values
  const normalized = paths.map(globalThis.AIKGPageviews.normalizePath)
  const formatted = globalThis.AIKGPageviews.formatCount(12345)

  // Then canonical paths and Chinese-locale grouping are returned
  assert.deepEqual(normalized, ["/", "/docs/", "/docs/"])
  assert.equal(formatted, "12,345")

  let inserted = null
  const heading = {
    insertAdjacentElement(position, element) {
      assert.equal(position, "afterend")
      inserted = element
    },
  }
  globalThis.location = { pathname: "/aikg/docs" }
  globalThis.document = {
    querySelector(selector) {
      if (selector === "[data-pageviews]") return inserted
      if (selector === ".md-content__inner > h1") return heading
      if (selector === 'script[src$="/assets/javascripts/pageviews.js"]') {
        return {
          src: "https://amoursec.github.io/aikg/assets/javascripts/pageviews.js",
        }
      }
      return null
    },
    createElement(tagName) {
      assert.equal(tagName, "p")
      return { className: "", dataset: {}, textContent: "" }
    },
  }
  let pageCounts = { "/docs/": 12345 }
  globalThis.fetch = async (url, options) => {
    assert.equal(url, "https://amoursec.github.io/aikg/assets/data/pageviews.json")
    assert.deepEqual(options, { cache: "no-store" })
    return {
      ok: true,
      async json() {
        return { pages: pageCounts }
      },
    }
  }

  // Given one matching article count
  // When rendering twice as instant navigation can do
  await globalThis.AIKGPageviews.render()
  const firstCounter = inserted
  await globalThis.AIKGPageviews.render()

  // Then one subdued counter is inserted immediately after the title
  assert.equal(inserted, firstCounter)
  assert.ok(inserted)
  assert.equal(inserted.className, "page-views")
  assert.equal(inserted.textContent, "浏览量：12,345")
  assert.ok(Object.hasOwn(inserted.dataset, "pageviews"))

  // Given an article without a snapshot entry
  // When rendering that article
  inserted = null
  globalThis.location = { pathname: "/aikg/missing" }

  // Then the page still exposes an explicit zero count
  await globalThis.AIKGPageviews.render()
  assert.ok(inserted)
  assert.equal(inserted.textContent, "浏览量：0")

  globalThis.location = { pathname: "/aikg/docs/" }
  for (const [pages, expected] of [
    [{ "/aikg/docs/": 7 }, "浏览量：7"],
    [{ "/docs/": 5, "/aikg/docs/": 7 }, "浏览量：12"],
  ]) {
    inserted = null
    pageCounts = pages
    await globalThis.AIKGPageviews.render()
    assert.equal(inserted.textContent, expected)
  }

  for (const invalid of [-1, 1.5, true, "7", null]) {
    inserted = null
    pageCounts = { "/docs/": 5, "/aikg/docs/": invalid }
    await globalThis.AIKGPageviews.render()
    assert.equal(inserted, null)
  }

  inserted = null
  globalThis.location = { pathname: "/docs/" }
  pageCounts = { "/docs/": 5 }
  await globalThis.AIKGPageviews.render()
  assert.equal(inserted.textContent, "浏览量：5")

  console.log("pageviews browser helpers and rendering: PASS")
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
