/**
 * main.js
 * Global UI interactions: Mobile Navigation Drawer & Accessibility setup.
 */

(function () {
    "use strict";

    document.addEventListener("DOMContentLoaded", () => {
        const mobileToggle = document.getElementById("mobile-menu-toggle");
        const mobileNav = document.getElementById("mobile-nav-drawer");

        if (mobileToggle && mobileNav) {
            mobileToggle.addEventListener("click", () => {
                const isOpen = mobileNav.classList.toggle("open");
                mobileToggle.setAttribute("aria-expanded", isOpen);
                mobileToggle.classList.toggle("active", isOpen);
            });

            // Close mobile menu when clicking outside
            document.addEventListener("click", (e) => {
                if (mobileNav.classList.contains("open") &&
                    !mobileNav.contains(e.target) &&
                    !mobileToggle.contains(e.target)) {
                    mobileNav.classList.remove("open");
                    mobileToggle.setAttribute("aria-expanded", "false");
                    mobileToggle.classList.remove("active");
                }
            });
        }
    });
})();
