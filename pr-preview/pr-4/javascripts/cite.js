/**
 * Client-side citation processor for MkDocs.
 *
 * Uses citation-js to resolve [@key] patterns in page content against
 * BibTeX files, rendering inline citations and bibliographies using CSL styles.
 *
 * Syntax:
 *   Inline citation:  [@key] or [@key1; @key2]       → (Author, Year)
 *   Full citation:    [!@key] or [!@key1; !@key2]    → full bibliography entry
 *   Bibliography:     <div class="bibliography"></div>
 *
 * Full citations ([!@key]) include a collapsible "Copy/Export Citation" section
 * with APA, MLA, Chicago, BibTeX, and RIS tabs, each with copy and save buttons.
 *
 * Configuration is read from a <script id="cite-config"> JSON block or defaults.
 */

(function () {
  'use strict'

  // --- Configuration ---

  const DEFAULTS = {
    bibFiles: ['assets/tidal.bib'],
    cslFile: 'assets/ieee.csl',
    template: 'ieee-custom',
    lang: 'en-US'
  }

  // Extra CSL files for format tabs (relative to site root)
  const EXTRA_CSL = {
    mla: 'assets/mla.csl',
    chicago: 'assets/chicago.csl'
  }

  function getConfig () {
    const el = document.getElementById('cite-config')
    if (el) {
      try {
        return Object.assign({}, DEFAULTS, JSON.parse(el.textContent))
      } catch (e) {
        console.warn('[cite] Failed to parse cite-config:', e)
      }
    }
    return DEFAULTS
  }

  // --- Helpers ---

  /** Resolve a path relative to the site root. */
  function siteRoot () {
    var segments = window.location.pathname.replace(/\/+$/, '').split('/').filter(Boolean)
    if (segments.length <= 1) return './'
    return '../'.repeat(segments.length - 1)
  }

  function resolveUrl (path) {
    if (path.startsWith('http://') || path.startsWith('https://')) return path
    var root = siteRoot()
    return new URL(path, new URL(root, window.location.href).href).href
  }

  async function fetchText (path) {
    const url = resolveUrl(path)
    const resp = await fetch(url)
    if (!resp.ok) throw new Error(`[cite] Failed to fetch ${url}: ${resp.status}`)
    return resp.text()
  }

  // --- Citation regex ---
  const CITE_RE = /\[@([a-zA-Z0-9_:.-]+(?:\s*;\s*@[a-zA-Z0-9_:.-]+)*)\]/g
  const FULL_CITE_RE = /\[!@([a-zA-Z0-9_:.-]+(?:\s*;\s*!@[a-zA-Z0-9_:.-]+)*)\]/g

  function parseCiteKeys (match) {
    return match.replace(/!?@/g, '').split(/\s*;\s*/).map(function (k) { return k.trim() })
  }

  /** Strip CSL-generated numbering from bibliography HTML. */
  function stripCslNumbering (html) {
    html = html.replace(/<div class="csl-left-margin">[\s\S]*?<\/div>/g, '')
    html = html.replace(/<div class="csl-right-inline">([\s\S]*?)<\/div>/g, '$1')
    return html
  }

  // --- BibTeX extraction ---

  /**
   * Parse raw BibTeX text into a map of citation key → raw BibTeX entry string.
   */
  function parseBibtexEntries (bibtexText) {
    var entries = {}
    var re = /@(\w+)\s*\{([^,]*),/g
    var match
    while ((match = re.exec(bibtexText)) !== null) {
      var type = match[1].toLowerCase()
      if (type === 'comment' || type === 'string' || type === 'preamble') continue
      var key = match[2].trim()
      var startIdx = match.index
      var braceDepth = 0
      var endIdx = match.index + match[0].length
      for (var i = match.index; i < bibtexText.length; i++) {
        if (bibtexText[i] === '{') braceDepth++
        else if (bibtexText[i] === '}') {
          braceDepth--
          if (braceDepth === 0) {
            endIdx = i + 1
            break
          }
        }
      }
      entries[key] = bibtexText.substring(startIdx, endIdx).trim()
    }
    return entries
  }

  // --- RIS generation from CSL-JSON ---

  /** Map CSL entry type to RIS TY tag */
  var CSL_TO_RIS_TYPE = {
    'article': 'JOUR',
    'article-journal': 'JOUR',
    'article-magazine': 'MGZN',
    'article-newspaper': 'NEWS',
    'book': 'BOOK',
    'chapter': 'CHAP',
    'dataset': 'DATA',
    'manuscript': 'UNPB',
    'paper-conference': 'CPAPER',
    'patent': 'PAT',
    'report': 'RPRT',
    'thesis': 'THES',
    'webpage': 'ELEC'
  }

  /**
   * Convert citation-js CSL-JSON data entries to RIS format string.
   * @param {object[]} cslData - array of CSL-JSON objects
   * @returns {string} RIS formatted text
   */
  function cslJsonToRis (cslData) {
    var records = cslData.map(function (entry) {
      var lines = []
      var risType = CSL_TO_RIS_TYPE[entry.type] || 'GEN'
      lines.push('TY  - ' + risType)

      // Authors
      if (entry.author) {
        entry.author.forEach(function (a) {
          if (a.literal) {
            lines.push('AU  - ' + a.literal)
          } else {
            var name = (a.family || '') + ', ' + (a.given || '')
            lines.push('AU  - ' + name.replace(/, $/, ''))
          }
        })
      }

      // Title
      if (entry.title) lines.push('TI  - ' + entry.title)

      // Journal / container
      if (entry['container-title']) lines.push('T2  - ' + entry['container-title'])

      // Volume, Issue, Pages
      if (entry.volume) lines.push('VL  - ' + entry.volume)
      if (entry.issue) lines.push('IS  - ' + entry.issue)
      if (entry.number) lines.push('IS  - ' + entry.number)
      if (entry.page) {
        var pages = String(entry.page).split(/[-–]/)
        lines.push('SP  - ' + pages[0].trim())
        if (pages.length > 1) lines.push('EP  - ' + pages[pages.length - 1].trim())
      }

      // Date
      if (entry.issued && entry.issued['date-parts'] && entry.issued['date-parts'][0]) {
        var dp = entry.issued['date-parts'][0]
        if (dp[0]) lines.push('PY  - ' + dp[0])
        if (dp[0]) {
          var da = String(dp[0])
          if (dp[1]) da += '/' + String(dp[1]).padStart(2, '0')
          if (dp[2]) da += '/' + String(dp[2]).padStart(2, '0')
          lines.push('DA  - ' + da)
        }
      }

      // DOI, URL
      if (entry.DOI) lines.push('DO  - ' + entry.DOI)
      if (entry.URL) lines.push('UR  - ' + entry.URL)

      // Publisher, Place
      if (entry.publisher) lines.push('PB  - ' + entry.publisher)
      if (entry['publisher-place']) lines.push('CY  - ' + entry['publisher-place'])

      // ISSN, ISBN
      if (entry.ISSN) lines.push('SN  - ' + entry.ISSN)
      if (entry.ISBN) lines.push('SN  - ' + entry.ISBN)

      // Abstract
      if (entry.abstract) lines.push('AB  - ' + entry.abstract)

      // Note
      if (entry.note) lines.push('N1  - ' + entry.note)

      lines.push('ER  - ')
      return lines.join('\n')
    })
    return records.join('\n\n')
  }

  // --- File download helper ---

  function downloadTextFile (text, filename) {
    var blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
    var url = URL.createObjectURL(blob)
    var a = document.createElement('a')
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  // --- SVG icons ---

  var ICON_CITE = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2H4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V20c0 1 0 1 1 1z"></path><path d="M15 21c3 0 7-1 7-8V5c0-1.25-.757-2.017-2-2h-4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2h.75c0 2.25.25 4-2.75 4v3c0 1 0 1 1 1z"></path></svg>'
  var WIDGET_STORAGE_KEY = 'cite-dataset-widget-selections'

  var ICON_COPY = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>'
  var ICON_CHECK = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>'
  var ICON_SAVE = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>'

  // --- Format tabs rendering ---

  /** Unique ID counter for tab groups */
  var tabGroupCounter = 0

  /** Keys whose format tabs should be expanded by default */
  var AUTO_EXPAND_KEYS = new Set(['mhkdr_submission'])

  /** File extension for each format */
  var FORMAT_EXT = {
    apa: '.txt',
    mla: '.txt',
    chicago: '.txt',
    bibtex: '.bib',
    ris: '.ris'
  }

  /**
   * Build HTML for the citation format tabs (collapsible details element).
   * @param {object} cite - citation-js Cite instance
   * @param {string[]} keys - citation keys to render
   * @param {object} bibtexEntries - key→raw BibTeX map
   * @param {object} extraTemplatesLoaded - which extra templates were successfully loaded
   * @param {string} lang - language code
   * @returns {string} HTML string
   */
  function buildFormatTabs (cite, keys, bibtexEntries, extraTemplatesLoaded, lang) {
    var groupId = 'cite-tab-group-' + (tabGroupCounter++)

    var formats = []

    if (extraTemplatesLoaded.apa) {
      formats.push({ id: 'apa', label: 'APA', template: 'apa-custom' })
    }
    if (extraTemplatesLoaded.mla) {
      formats.push({ id: 'mla', label: 'MLA', template: 'mla' })
    }
    if (extraTemplatesLoaded.chicago) {
      formats.push({ id: 'chicago', label: 'Chicago', template: 'chicago' })
    }
    formats.push({ id: 'bibtex', label: 'BibTeX', template: null })
    formats.push({ id: 'ris', label: 'RIS (Zotero)', template: null })

    if (formats.length === 0) return ''

    // Build filename base from first key
    var fileBase = keys.length === 1 ? keys[0] : 'citations'

    var tabsHtml = '<div class="cite-format-tabs" role="tablist">'
    var contentsHtml = ''

    formats.forEach(function (fmt, idx) {
      var isActive = idx === 0
      var tabId = groupId + '-tab-' + fmt.id
      var panelId = groupId + '-panel-' + fmt.id

      tabsHtml += '<button class="cite-format-tab' + (isActive ? ' active' : '') + '"' +
        ' role="tab"' +
        ' aria-selected="' + isActive + '"' +
        ' aria-controls="' + panelId + '"' +
        ' id="' + tabId + '"' +
        ' data-format="' + fmt.id + '">' +
        fmt.label + '</button>'

      var contentText = ''
      var isMono = fmt.id === 'bibtex' || fmt.id === 'ris'

      if (fmt.id === 'bibtex') {
        var bibtexParts = keys.map(function (key) {
          return bibtexEntries[key] || '% BibTeX entry not found for ' + key
        })
        contentText = bibtexParts.join('\n\n')
      } else if (fmt.id === 'ris') {
        // Generate RIS from CSL-JSON data
        try {
          var cslData = keys.map(function (key) {
            return cite.data.find(function (d) { return d.id === key })
          }).filter(Boolean)
          contentText = cslJsonToRis(cslData)
        } catch (e) {
          console.warn('[cite] Error generating RIS:', e)
          contentText = 'Error generating RIS format.'
        }
      } else {
        // APA, MLA, Chicago — join with double newline for readability
        try {
          var parts = keys.map(function (key) {
            var text = cite.format('bibliography', {
              template: fmt.template,
              lang: lang,
              format: 'text',
              entry: [key]
            })
            return text.trim()
          })
          contentText = parts.join('\n\n')
        } catch (e) {
          console.warn('[cite] Error formatting ' + fmt.label + ':', e)
          contentText = 'Error generating ' + fmt.label + ' citation.'
        }
      }

      // Escape HTML for safe insertion
      var escapedText = contentText
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')

      var downloadFilename = fileBase + FORMAT_EXT[fmt.id]

      if (fmt.id === 'ris') {
        // RIS: download-only, no code preview
        contentsHtml += '<div class="cite-format-content"' +
          (isActive ? ' ' : '') +
          ' role="tabpanel"' +
          ' id="' + panelId + '"' +
          ' aria-labelledby="' + tabId + '"' +
          (isActive ? '' : ' hidden') + '>' +
          '<div class="cite-action-bar">' +
          '<button class="cite-save-btn" data-save-text="' + escapedText + '" data-filename="' + downloadFilename + '" title="Download RIS file">' +
          ICON_SAVE + ' Download .ris</button>' +
          '</div>' +
          '</div>'
      } else {
        // APA, MLA, Chicago, BibTeX: code preview with inline copy button
        contentsHtml += '<div class="cite-format-content' + (isMono ? ' cite-mono' : '') +
          (isActive ? ' active' : '') + '"' +
          ' role="tabpanel"' +
          ' id="' + panelId + '"' +
          ' aria-labelledby="' + tabId + '"' +
          (isActive ? '' : ' hidden') + '>' +
          '<div class="cite-format-pre-wrapper">' +
          '<pre class="cite-format-pre"><code>' + escapedText + '</code></pre>' +
          '<button class="cite-copy-btn cite-copy-inline" data-copy-text="' + escapedText + '" title="Copy to clipboard">' +
          ICON_COPY + '</button>' +
          '</div>' +
          '<div class="cite-action-bar">' +
          '<button class="cite-save-btn" data-save-text="' + escapedText + '" data-filename="' + downloadFilename + '" title="Save as file">' +
          ICON_SAVE + ' Save</button>' +
          '</div>' +
          '</div>'
      }
    })

    tabsHtml += '</div>'

    var shouldExpand = keys.some(function (k) { return AUTO_EXPAND_KEYS.has(k) })

    var chevron = '<span class="cite-chevron"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg></span>'

    return '<details class="cite-formats"' + (shouldExpand ? ' open' : '') + '>' +
      '<summary>Copy/Export Citation' + chevron + '</summary>' +
      '<div class="cite-formats-inner">' +
      tabsHtml +
      contentsHtml +
      '</div>' +
      '</details>'
  }

  // --- Tab switching + copy + save event delegation ---

  function initTabBehavior (container) {
    container.addEventListener('click', function (e) {
      // Tab switching
      var tab = e.target.closest('.cite-format-tab')
      if (tab) {
        var tabBar = tab.parentElement
        var wrapper = tabBar.parentElement

        tabBar.querySelectorAll('.cite-format-tab').forEach(function (t) {
          t.classList.remove('active')
          t.setAttribute('aria-selected', 'false')
        })
        wrapper.querySelectorAll('.cite-format-content').forEach(function (p) {
          p.classList.remove('active')
          p.hidden = true
        })

        tab.classList.add('active')
        tab.setAttribute('aria-selected', 'true')
        var panelId = tab.getAttribute('aria-controls')
        var panel = document.getElementById(panelId)
        if (panel) {
          panel.classList.add('active')
          panel.hidden = false
        }
        return
      }

      // Copy button
      var copyBtn = e.target.closest('.cite-copy-btn')
      if (copyBtn) {
        var text = copyBtn.dataset.copyText
        var textarea = document.createElement('textarea')
        textarea.innerHTML = text
        var plainText = textarea.value

        navigator.clipboard.writeText(plainText).then(function () {
          var origHtml = copyBtn.innerHTML
          copyBtn.innerHTML = ICON_CHECK + ' Copied!'
          copyBtn.classList.add('copied')
          setTimeout(function () {
            copyBtn.innerHTML = origHtml
            copyBtn.classList.remove('copied')
          }, 2000)
        }).catch(function (err) {
          console.warn('[cite] Copy failed:', err)
        })
        return
      }

      // Save button
      var saveBtn = e.target.closest('.cite-save-btn')
      if (saveBtn) {
        var saveText = saveBtn.dataset.saveText
        var textarea2 = document.createElement('textarea')
        textarea2.innerHTML = saveText
        var plainSaveText = textarea2.value
        var filename = saveBtn.dataset.filename || 'citation.txt'

        downloadTextFile(plainSaveText, filename)

        var origSaveHtml = saveBtn.innerHTML
        saveBtn.innerHTML = ICON_CHECK + ' Saved!'
        saveBtn.classList.add('saved')
        setTimeout(function () {
          saveBtn.innerHTML = origSaveHtml
          saveBtn.classList.remove('saved')
        }, 2000)
      }
    })
  }

  // --- Cite Dataset Widget ---

  /**
   * Find all BibTeX keys that start with a given location prefix.
   * E.g. prefix "ak_cook" matches "ak_cook_deb2025_characterizing".
   */
  function findKeysForPrefix (prefix, allKeys) {
    var lowerPrefix = prefix.toLowerCase() + '_'
    return allKeys.filter(function (k) {
      return k.toLowerCase().indexOf(lowerPrefix) === 0
    })
  }

  /**
   * Build and wire the "Cite this dataset" widget.
   */
  function buildCiteWidget (widgetEl, cite, availableKeys, bibtexEntries, extraTemplatesLoaded, lang) {
    var locationsJson
    try {
      locationsJson = JSON.parse(widgetEl.getAttribute('data-locations'))
    } catch (e) {
      console.warn('[cite] Failed to parse widget data-locations:', e)
      return
    }
    if (!locationsJson || typeof locationsJson !== 'object') return

    var allKeys = Array.from(availableKeys)

    // Build prefix → { label, keys } mapping, only for prefixes with actual bib entries
    var locationEntries = []
    Object.keys(locationsJson).forEach(function (prefix) {
      var keys = findKeysForPrefix(prefix, allKeys)
      if (keys.length > 0) {
        locationEntries.push({ prefix: prefix, label: locationsJson[prefix], keys: keys })
      }
    })

    // Read persisted selections from localStorage
    var savedSelections = {}
    try {
      var stored = localStorage.getItem(WIDGET_STORAGE_KEY)
      if (stored) savedSelections = JSON.parse(stored)
    } catch (e) { /* ignore */ }

    function saveSelections () {
      var selections = {}
      locationEntries.forEach(function (loc) {
        var cb = widgetEl.querySelector('input[data-prefix="' + loc.prefix + '"]')
        if (cb) selections[loc.prefix] = cb.checked
      })
      try {
        localStorage.setItem(WIDGET_STORAGE_KEY, JSON.stringify(selections))
      } catch (e) { /* ignore */ }
    }

    var widgetTitle = widgetEl.getAttribute('data-title') || 'Dataset'

    var chevron = '<span class="cite-chevron"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg></span>'

    var html = '<details class="cite-widget-details" open>' +
      '<summary class="cite-widget-summary">' +
      'How to Cite the ' + widgetTitle + chevron +
      '</summary>' +
      '<div class="cite-widget-body">'

    // Location checkboxes (shown first)
    if (locationEntries.length > 0) {
      html += '<div class="cite-widget-locations">' +
        '<div class="cite-widget-locations-label">Locations used in your work:</div>' +
        '<div class="cite-widget-select-all-row">' +
        '<label class="cite-widget-checkbox-label">' +
        '<input type="checkbox" id="cite-widget-select-all"> Select all' +
        '</label></div>' +
        '<div class="cite-widget-location-grid">'

      locationEntries.forEach(function (loc) {
        var checked = savedSelections[loc.prefix] === true ? ' checked' : ''
        html += '<label class="cite-widget-checkbox-label">' +
          '<input type="checkbox" data-prefix="' + loc.prefix + '"' + checked + '> ' +
          loc.label +
          '</label>'
      })

      html += '</div>'

      html += '<p class="cite-widget-hint">A dataset-level citation is included automatically. ' +
        'Location selections add the relevant model-validation references.</p>'
      html += '</div>'
    }

    // Tabs container
    html += '<div class="cite-widget-tabs-container"></div>'
    html += '</div></details>'

    widgetEl.innerHTML = html

    var tabsContainer = widgetEl.querySelector('.cite-widget-tabs-container')
    var selectAllCb = widgetEl.querySelector('#cite-widget-select-all')

    function getSelectedKeys () {
      var keys = ['mhkdr_submission']
      locationEntries.forEach(function (loc) {
        var cb = widgetEl.querySelector('input[data-prefix="' + loc.prefix + '"]')
        if (cb && cb.checked) {
          loc.keys.forEach(function (k) {
            if (keys.indexOf(k) === -1) keys.push(k)
          })
        }
      })
      return keys
    }

    function updateSelectAll () {
      if (!selectAllCb) return
      var cbs = widgetEl.querySelectorAll('input[data-prefix]')
      var total = cbs.length
      var checked = 0
      cbs.forEach(function (cb) { if (cb.checked) checked++ })
      selectAllCb.checked = checked === total
      selectAllCb.indeterminate = checked > 0 && checked < total
    }

    function renderTabs () {
      var keys = getSelectedKeys()
      var tabsHtml = buildFormatTabs(cite, keys, bibtexEntries, extraTemplatesLoaded, lang)
      tabsContainer.innerHTML = tabsHtml

      // Force the inner <details> open and hide its <summary>
      var innerDetails = tabsContainer.querySelector('.cite-formats')
      if (innerDetails) {
        innerDetails.open = true
        var innerSummary = innerDetails.querySelector('summary')
        if (innerSummary) innerSummary.style.display = 'none'
      }
    }

    // Wire up checkbox handlers
    var locationCbs = widgetEl.querySelectorAll('input[data-prefix]')
    locationCbs.forEach(function (cb) {
      cb.addEventListener('change', function () {
        saveSelections()
        updateSelectAll()
        renderTabs()
      })
    })

    if (selectAllCb) {
      selectAllCb.addEventListener('change', function () {
        var isChecked = selectAllCb.checked
        locationCbs.forEach(function (cb) { cb.checked = isChecked })
        saveSelections()
        renderTabs()
      })
    }

    // Initial render
    updateSelectAll()
    renderTabs()
  }

  // --- Interactive citation tooltips ---

  function initCiteTooltips (container, keyToTooltip) {
    var tooltip = document.createElement('div')
    tooltip.className = 'cite-tooltip'
    // Invisible bridge so mouse can travel from link to tooltip
    tooltip.innerHTML = '<div class="cite-tooltip-bridge"></div><div class="cite-tooltip-body"></div>'
    document.body.appendChild(tooltip)

    var body = tooltip.querySelector('.cite-tooltip-body')
    var hideTimer = null
    var currentLink = null

    function show (link) {
      var key = link.getAttribute('data-cite-key')
      if (!key || !keyToTooltip[key]) return

      clearTimeout(hideTimer)
      currentLink = link
      body.innerHTML = keyToTooltip[key]
      // Make all links in the tooltip open in a new tab
      body.querySelectorAll('a').forEach(function (a) {
        a.setAttribute('target', '_blank')
        a.setAttribute('rel', 'noopener noreferrer')
      })
      tooltip.classList.add('visible')

      // Position above the link
      var rect = link.getBoundingClientRect()
      var tooltipRect

      // Temporarily make visible off-screen to measure
      tooltip.style.left = '-9999px'
      tooltip.style.top = '-9999px'
      tooltipRect = tooltip.getBoundingClientRect()

      var left = rect.left + rect.width / 2 - tooltipRect.width / 2
      var top = rect.top + window.scrollY - tooltipRect.height - 8

      // Keep within viewport horizontally
      var margin = 12
      if (left < margin) left = margin
      if (left + tooltipRect.width > window.innerWidth - margin) {
        left = window.innerWidth - margin - tooltipRect.width
      }

      // If no room above, show below
      if (rect.top - tooltipRect.height - 8 < 0) {
        top = rect.bottom + window.scrollY + 8
        tooltip.classList.add('cite-tooltip-below')
      } else {
        tooltip.classList.remove('cite-tooltip-below')
      }

      tooltip.style.left = left + 'px'
      tooltip.style.top = top + 'px'
    }

    function scheduleHide () {
      hideTimer = setTimeout(function () {
        tooltip.classList.remove('visible')
        currentLink = null
      }, 150)
    }

    function cancelHide () {
      clearTimeout(hideTimer)
    }

    // Hover on cite links
    container.addEventListener('mouseenter', function (e) {
      var link = e.target.closest('.cite-link[data-cite-key]')
      if (link) show(link)
    }, true)

    container.addEventListener('mouseleave', function (e) {
      var link = e.target.closest('.cite-link[data-cite-key]')
      if (link) scheduleHide()
    }, true)

    // Keep tooltip open while hovering over it
    tooltip.addEventListener('mouseenter', cancelHide)
    tooltip.addEventListener('mouseleave', scheduleHide)
  }

  // --- Main ---

  async function processCitations () {
    var Cite
    try {
      Cite = require('citation-js')
    } catch (e) {
      Cite = window.Cite
    }
    if (!Cite) {
      console.error('[cite] citation-js not loaded')
      return
    }

    const config = getConfig()
    const content = document.querySelector('.md-content')
    if (!content) return

    // Check if there are any citations on this page
    const html = content.innerHTML
    var hasCites = CITE_RE.test(html)
    CITE_RE.lastIndex = 0
    var hasFullCites = FULL_CITE_RE.test(html)
    FULL_CITE_RE.lastIndex = 0
    var hasWidget = !!content.querySelector('#cite-dataset-widget')
    if (!hasCites && !hasFullCites && !content.querySelector('.bibliography') && !hasWidget) {
      document.body.classList.add('cite-ready')
      return
    }

    // Immediately replace raw citation text with styled placeholders
    // so the user never sees unstyled [@key] brackets
    if (hasFullCites) {
      FULL_CITE_RE.lastIndex = 0
      content.innerHTML = content.innerHTML.replace(FULL_CITE_RE, function () {
        return '<span class="cite-placeholder-block"></span>'
      })
    }
    if (hasCites) {
      CITE_RE.lastIndex = 0
      content.innerHTML = content.innerHTML.replace(CITE_RE, function () {
        return '<span class="cite-placeholder">\u00a0\u00a0\u00a0</span>'
      })
    }

    document.body.classList.add('cite-loading')

    // Load all bib files
    let allBibtex = ''
    for (const bibFile of config.bibFiles) {
      try {
        const bib = await fetchText(bibFile)
        allBibtex += '\n' + bib
      } catch (e) {
        console.warn('[cite] Could not load bib file:', bibFile, e)
      }
    }

    if (!allBibtex.trim()) {
      console.warn('[cite] No bibliography data loaded')
      document.body.classList.add('cite-ready')
      return
    }

    // Parse BibTeX entries for the BibTeX tab
    var bibtexEntries = parseBibtexEntries(allBibtex)

    // Parse all references
    let cite
    try {
      cite = new Cite(allBibtex)
    } catch (e) {
      console.error('[cite] Failed to parse bibliography:', e)
      document.body.classList.add('cite-ready')
      return
    }

    // Load primary CSL (IEEE)
    if (config.cslFile) {
      try {
        const cslXml = await fetchText(config.cslFile)
        const cslConfig = Cite.plugins.config.get('@csl')
        cslConfig.templates.add(config.template, cslXml)
      } catch (e) {
        console.warn('[cite] Could not load CSL file, falling back to APA:', e)
        config.template = 'apa'
      }
    }

    // Load extra CSL templates for format tabs
    var extraTemplatesLoaded = { apa: false, mla: false, chicago: false }
    var cslConfig = Cite.plugins.config.get('@csl')

    // APA
    try {
      var apaCsl = await fetchText('assets/apa.csl')
      cslConfig.templates.add('apa-custom', apaCsl)
      extraTemplatesLoaded.apa = true
    } catch (e) {
      console.warn('[cite] Could not load APA CSL:', e)
    }

    // MLA
    try {
      var mlaCsl = await fetchText(EXTRA_CSL.mla)
      cslConfig.templates.add('mla', mlaCsl)
      extraTemplatesLoaded.mla = true
    } catch (e) {
      console.warn('[cite] Could not load MLA CSL:', e)
    }

    // Chicago
    try {
      var chicagoCsl = await fetchText(EXTRA_CSL.chicago)
      cslConfig.templates.add('chicago', chicagoCsl)
      extraTemplatesLoaded.chicago = true
    } catch (e) {
      console.warn('[cite] Could not load Chicago CSL:', e)
    }

    // Build a lookup of all available keys
    const availableKeys = new Set(cite.getIds())

    // Collect all cited keys on this page (in order of appearance)
    // Use the original html (saved before placeholders were inserted)
    const citedKeys = []
    const seenKeys = new Set()
    let match

    CITE_RE.lastIndex = 0
    while ((match = CITE_RE.exec(html)) !== null) {
      const keys = parseCiteKeys(match[1])
      for (const key of keys) {
        if (!seenKeys.has(key)) { seenKeys.add(key); citedKeys.push(key) }
      }
    }
    FULL_CITE_RE.lastIndex = 0
    while ((match = FULL_CITE_RE.exec(html)) !== null) {
      const keys = parseCiteKeys(match[1])
      for (const key of keys) {
        if (!seenKeys.has(key)) { seenKeys.add(key); citedKeys.push(key) }
      }
    }

    // Build key → number map and key → plain text citation for tooltips
    var keyToNum = {}
    var keyToTooltip = {}
    var validCitedKeys = citedKeys.filter(function (k) { return availableKeys.has(k) })
    validCitedKeys.forEach(function (key, i) { keyToNum[key] = i + 1 })

    // Pre-compute tooltip HTML for each cited key (with clickable DOI links)
    validCitedKeys.forEach(function (key) {
      try {
        var html = cite.format('bibliography', {
          template: config.template,
          lang: config.lang,
          format: 'html',
          entry: [key],
          hyperlinks: true
        }).trim()
        html = stripCslNumbering(html)
        html = html.replace(/<\/?div[^>]*>/g, '').trim()
        keyToTooltip[key] = html
      } catch (e) {
        keyToTooltip[key] = key
      }
    })

    // Detect numeric vs author-year style
    var isNumericStyle = false
    if (validCitedKeys.length > 0) {
      try {
        var testCite = cite.format('citation', {
          template: config.template, lang: config.lang, format: 'text',
          entry: [validCitedKeys[0]]
        }).trim()
        isNumericStyle = /^\[\d+\]$/.test(testCite)
      } catch (e) { /* assume non-numeric */ }
    }

    // Reset tab group counter for this page
    tabGroupCounter = 0

    // Restore original HTML so regex replacements can find [@key] patterns
    content.innerHTML = html

    // Replace [!@key] with full bibliography-style citations + format tabs
    FULL_CITE_RE.lastIndex = 0
    content.innerHTML = content.innerHTML.replace(FULL_CITE_RE, function (fullMatch, inner) {
      const keys = parseCiteKeys(inner)
      const vKeys = keys.filter(function (k) { return availableKeys.has(k) })

      if (vKeys.length === 0) {
        console.warn('[cite] Unknown citation key(s):', keys)
        return '<span class="cite-error" title="Unknown citation key">' + fullMatch + '</span>'
      }

      try {
        var parts = vKeys.map(function (key) {
          var h = cite.format('bibliography', {
            template: config.template,
            lang: config.lang,
            format: 'html',
            entry: [key],
            hyperlinks: true
          })
          return stripCslNumbering(h)
        })

        var mainCitation = '<div class="cite-full">' + parts.join('') + '</div>'
        var formatTabs = buildFormatTabs(cite, vKeys, bibtexEntries, extraTemplatesLoaded, config.lang)

        return mainCitation + formatTabs
      } catch (e) {
        console.warn('[cite] Error formatting full citation:', keys, e)
        return fullMatch
      }
    })

    // Replace [@key] with short inline citations
    CITE_RE.lastIndex = 0
    content.innerHTML = content.innerHTML.replace(CITE_RE, function (fullMatch, inner) {
      const keys = parseCiteKeys(inner)
      const vKeys = keys.filter(function (k) { return availableKeys.has(k) })

      if (vKeys.length === 0) {
        console.warn('[cite] Unknown citation key(s):', keys)
        return '<span class="cite-error" title="Unknown citation key">' + fullMatch + '</span>'
      }

      if (isNumericStyle) {
        var links = vKeys.map(function (k) {
          var num = keyToNum[k]
          return '<a href="#cite-ref-' + k + '" class="cite-link" data-cite-key="' + k + '">' + num + '</a>'
        })
        return '<span class="cite-inline">[' + links.join(', ') + ']</span>'
      }

      try {
        const inlineCite = cite.format('citation', {
          template: config.template,
          lang: config.lang,
          format: 'html',
          entry: vKeys
        })
        var firstKey = vKeys[0]
        return '<a href="#cite-ref-' + firstKey + '" class="cite-inline cite-link" data-cite-key="' + firstKey + '">' + inlineCite + '</a>'
      } catch (e) {
        console.warn('[cite] Error formatting citation:', keys, e)
        return fullMatch
      }
    })

    // Render bibliography / references list
    if (validCitedKeys.length > 0) {
      try {
        var listItems = validCitedKeys.map(function (key) {
          var entry = cite.format('bibliography', {
            template: config.template,
            lang: config.lang,
            format: 'html',
            entry: [key],
            hyperlinks: true
          })
          entry = stripCslNumbering(entry)
          entry = entry.replace(/<\/?div[^>]*>/g, '').trim()
          return '<li id="cite-ref-' + key + '">' + entry + '</li>'
        })
        var bibHtml = '<ol class="cite-bib-list">' + listItems.join('\n') + '</ol>'

        // Use existing .bibliography div if present, otherwise auto-inject
        // Make all bibliography links open in a new tab
        var bibTemp = document.createElement('div')
        bibTemp.innerHTML = bibHtml
        bibTemp.querySelectorAll('a').forEach(function (a) {
          a.setAttribute('target', '_blank')
          a.setAttribute('rel', 'noopener noreferrer')
        })
        bibHtml = bibTemp.innerHTML

        var bibDivs = content.querySelectorAll('.bibliography')
        if (bibDivs.length > 0) {
          content.classList.add('cite-bib-page')
          bibDivs.forEach(function (div) {
            div.innerHTML = bibHtml
            div.classList.add('cite-bibliography')
          })
        } else {
          // Auto-inject a References section at the end of the content
          // Insert before the cite-dataset-widget if present, otherwise append
          var refsSection = document.createElement('div')
          refsSection.className = 'cite-bibliography cite-bibliography-auto'
          refsSection.innerHTML = '<h2 id="references">References' +
            '<a class="headerlink" href="#references" title="Permanent link">¶</a></h2>' +
            bibHtml

          var widget = content.querySelector('.cite-dataset-widget')
          var contentInner = content.querySelector('.md-content__inner') || content
          if (widget) {
            contentInner.insertBefore(refsSection, widget)
          } else {
            contentInner.appendChild(refsSection)
          }
        }
      } catch (e) {
        console.error('[cite] Error formatting bibliography:', e)
      }
    }

    // Initialize cite dataset widget if present
    var widgetEl = content.querySelector('#cite-dataset-widget')
    if (widgetEl) {
      buildCiteWidget(widgetEl, cite, availableKeys, bibtexEntries, extraTemplatesLoaded, config.lang)
    }

    // Initialize tab switching and copy/save behavior
    initTabBehavior(content)

    // Initialize interactive citation tooltips
    initCiteTooltips(content, keyToTooltip)

    // Signal that citations are processed
    document.body.classList.remove('cite-loading')
    document.body.classList.add('cite-ready')

    // Re-run MathJax if present
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise()
    }
  }

  // --- Integration with MkDocs Material instant loading ---

  if (typeof document$ !== 'undefined') {
    document$.subscribe(function () {
      document.body.classList.remove('cite-ready', 'cite-loading')
      processCitations()
    })
  } else {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', processCitations)
    } else {
      processCitations()
    }
  }
})()
