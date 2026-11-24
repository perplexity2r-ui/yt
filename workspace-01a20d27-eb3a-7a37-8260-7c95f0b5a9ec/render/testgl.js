const puppeteer = require('puppeteer');
(async ()=>{
  const browser = await puppeteer.launch({
    headless: 'shell'===0?false:true,
    args: ['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--disable-dev-shm-usage']
  });
  const page = await browser.newPage();
  await page.setContent(`<canvas id=c></canvas><script>
    const gl = document.getElementById('c').getContext('webgl2');
    window.info = gl ? gl.getParameter(gl.VERSION)+' | '+gl.getParameter(gl.RENDERER) : 'NO WEBGL';
  </script>`);
  console.log(await page.evaluate(()=>window.info));
  await browser.close();
})();
