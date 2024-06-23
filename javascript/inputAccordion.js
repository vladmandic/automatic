function inputAccordionChecked(id, checked) {
  const accordion = gradioApp().getElementById(id);
  accordion.visibleCheckbox.checked = checked;
  accordion.onVisibleCheckboxChange();
}

function setupAccordion(accordion) {
  const labelWrap = accordion.querySelector('.label-wrap');
  const gradioCheckbox = gradioApp().querySelector(`#${accordion.id}-checkbox input`);
  const extra = gradioApp().querySelector(`#${accordion.id}-extra`);
  const span = labelWrap.querySelector('span');
  let linked = true;
  const isOpen = () => labelWrap.classList.contains('open');
  const observerAccordionOpen = new MutationObserver((mutations) => {
    mutations.forEach((mutationRecord) => {
      accordion.classList.toggle('input-accordion-open', isOpen());
      if (linked) {
        accordion.visibleCheckbox.checked = isOpen();
        accordion.onVisibleCheckboxChange();
      }
    });
  });
  observerAccordionOpen.observe(labelWrap, { attributes: true, attributeFilter: ['class'] });
  if (extra) labelWrap.insertBefore(extra, labelWrap.lastElementChild);
  accordion.onChecked = (checked) => {
    if (isOpen() !== checked) labelWrap.click();
  };

  const visibleCheckbox = document.createElement('INPUT');
  visibleCheckbox.type = 'checkbox';
  visibleCheckbox.checked = isOpen();
  visibleCheckbox.id = `${accordion.id}-visible-checkbox`;
  visibleCheckbox.className = `${gradioCheckbox.className} input-accordion-checkbox`;
  span.insertBefore(visibleCheckbox, span.firstChild);
  accordion.visibleCheckbox = visibleCheckbox;
  accordion.onVisibleCheckboxChange = () => {
    if (linked && isOpen() !== visibleCheckbox.checked) labelWrap.click();
    gradioCheckbox.checked = visibleCheckbox.checked;
    updateInput(gradioCheckbox);
  };

  visibleCheckbox.addEventListener('click', (event) => {
    linked = false;
    event.stopPropagation();
  });
  visibleCheckbox.addEventListener('input', accordion.onVisibleCheckboxChange);
}

// onUiLoaded(() => {
//  for (const accordion of gradioApp().querySelectorAll('.input-accordion')) setupAccordion(accordion);
// });

function initAccordions() {
  for (const accordion of gradioApp().querySelectorAll('.input-accordion')) setupAccordion(accordion);
}
