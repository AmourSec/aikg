"use strict"

const assert = require("node:assert/strict")
const fs = require("node:fs")
const path = require("node:path")
const vm = require("node:vm")

const source = fs.readFileSync(
  path.join(__dirname, "../docs/assets/javascripts/analytics.js"),
  "utf8",
)

function runTracker(hostname) {
  let appended = null
  const context = {
    location: { hostname },
    document: {
      createElement(tagName) {
        assert.equal(tagName, "script")
        return { dataset: {} }
      },
      head: {
        appendChild(element) {
          appended = element
        },
      },
    },
  }

  vm.runInNewContext(source, context)
  return appended
}

// Given the public GitHub Pages hostname
// When the analytics loader runs
const productionTracker = runTracker("amoursec.github.io")

// Then it loads the GoatCounter endpoint for the configured public site code
assert.equal(productionTracker.async, true)
assert.equal(productionTracker.src, "https://gc.zgo.at/count.js")
assert.equal(
  productionTracker.dataset.goatcounter,
  "https://amoursec.goatcounter.com/count",
)

// Given a local preview hostname
// When the analytics loader runs
const localTracker = runTracker("localhost")

// Then no analytics request is registered
assert.equal(localTracker, null)

console.log("GoatCounter analytics loader: PASS")
