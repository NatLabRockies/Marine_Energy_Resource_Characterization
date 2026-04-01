// Custom scroll-spy: highlight the TOC link for the section currently in view
(function () {
  var OFFSET = 100; // px from top of viewport to consider a heading "active"

  function run() {
    var tocLinks = document.querySelectorAll(".md-nav--secondary .md-nav__link");
    if (!tocLinks.length) return;

    // Build a map from anchor id -> TOC link element
    var linkMap = {};
    tocLinks.forEach(function (link) {
      var href = link.getAttribute("href");
      if (href && href.startsWith("#")) {
        linkMap[href.slice(1)] = link;
      }
    });

    var headingIds = Object.keys(linkMap);
    if (!headingIds.length) return;

    function update() {
      var active = null;

      // Find the last heading that has scrolled past the offset line
      for (var i = 0; i < headingIds.length; i++) {
        var el = document.getElementById(headingIds[i]);
        if (el && el.getBoundingClientRect().top < OFFSET) {
          active = headingIds[i];
        }
      }

      // Clear all active states set by us and by mkdocs-material
      tocLinks.forEach(function (link) {
        link.classList.remove("md-nav__link--active");
        link.classList.remove("toc-active");
        link.setAttribute("data-toc-active", "false");
      });

      // Set active on the correct link
      if (active && linkMap[active]) {
        linkMap[active].classList.add("toc-active");
        linkMap[active].setAttribute("data-toc-active", "true");
      }
    }

    // Override mkdocs-material's scroll-spy by repeatedly clearing its classes
    var observer = new MutationObserver(function () {
      tocLinks.forEach(function (link) {
        if (
          link.classList.contains("md-nav__link--active") &&
          link.getAttribute("data-toc-active") !== "true"
        ) {
          link.classList.remove("md-nav__link--active");
        }
      });
    });

    // Watch the TOC container for class changes
    var tocContainer = document.querySelector(".md-nav--secondary");
    if (tocContainer) {
      observer.observe(tocContainer, {
        subtree: true,
        attributes: true,
        attributeFilter: ["class"],
      });
    }

    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update, { passive: true });
    update();
  }

  // Run after DOM is ready and after a short delay for mkdocs-material to initialize
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      setTimeout(run, 100);
    });
  } else {
    setTimeout(run, 100);
  }
})();
