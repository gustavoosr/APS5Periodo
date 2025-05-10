function openMenu() {
    document.getElementById("menuSide").style.width = "250px";
    document.getElementById("content").style.marginRight = "250px";
}

function closeMenu() {
    document.getElementById("menuSide").style.width = "0";
    document.getElementById("content").style.marginRight = "0";
}
function redirect() {
    setTimeout(() => {
        window.location.href = 'ChatBot.html';
    }, 2000); // 2000ms (2 segundos)
}
function redirectGraphic() {
    setTimeout(() => {
        window.location.href = 'Grafico.html';
    }, 2000); // 2000ms (2 segundos)
}

function openTrafficMonitor(){
    fetch("http://localhost:3000/network-logs")
    .then(res => res.json())
    .then(logs => {
        const tbody = document.getElementById("trafficTable").querySelector("tbody");
        tbody.innerHTML = logs.map(log => `
          <tr>
            <td>${log.timestamp}</td>
            <td>${log.method}</td>
            <td>${log.path}</td>
            <td>${log.status}</td>
            <td>${log.protocol}</td>
            <td>${log.duration}</td>
          </tr>
        `).join("");
        document.getElementById("trafficPopup").classList.remove("hidden");
      })
      .catch(console.error("Erro ao carregar tráfego:"));
  }
  function closeTrafficPopup() {
    document.getElementById("trafficPopup").classList.add("hidden");
  }