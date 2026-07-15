// Show the "back to top" button once the user has scrolled past the header.
const backToTop = document.querySelector(".back-to-top");

if (backToTop) {
  const toggleVisibility = () => {
    if (window.scrollY > 480) {
      backToTop.classList.add("is-visible");
    } else {
      backToTop.classList.remove("is-visible");
    }
  };

  toggleVisibility();
  window.addEventListener("scroll", toggleVisibility, { passive: true });
}
