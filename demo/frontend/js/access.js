/* Automatic logoff. The page polls and streams on its own, so the server cannot tell an attended screen from an
 * abandoned one; this watches for the user instead and signs out after the deployment's idle time. Zero minutes means
 * sign-in is off (the local demo). */
(function () {
  const minutes = Number(document.body.dataset.idleMinutes) || 0;
  if (!minutes) return;
  let timer = null;
  const reset = () => {
    clearTimeout(timer);
    timer = setTimeout(() => window.location.assign('/logout'), minutes * 60 * 1000);
  };
  ['pointerdown', 'pointermove', 'keydown', 'wheel', 'touchstart'].forEach((type) =>
    window.addEventListener(type, reset, { passive: true }));
  reset();
})();
