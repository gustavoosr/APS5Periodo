let newsJson = [];
window.addEventListener("DOMContentLoaded", () => {
  fetch("/DataSet/dataSetMeioAmbiente.json")
    .then(Response => Response.json())
    .then(data => {
      newsJson = data;
      const newsFiltered = filterLast5Days(data);
      generateTableAndGraphic(newsFiltered);
    })
    .catch(error => console.error("Erro ao carregar Json", error));
});

function changeFilterDate() {
  const filter = document.getElementById("filterDate");
  filter.classList.toggle("visible");
  filter.classList.toggle("hidden");
}

function generateTableAndGraphic(filteredData) {
  const resume = {};

  if (filteredData.length === 0) {
    document.getElementById("sectionTable").innerHTML = "<p>Nenhuma notícia encontrada nos últimos 5 dias :/ .</p>";
    document.getElementById("sectionGraphic").innerHTML = "";
    return;
  }
  filteredData.forEach(newsJson => {
    const subject = (newsJson.assunto || "Outros").trim();
    let classification = (newsJson.classificacão || "").trim().toLowerCase();
    // Corrigir para a forma esperada pelo gráfico (primeira maiúscula)
    if (classification === "boa" || classification === "ruim" || classification === "irrelevante") {
      classification = classification.charAt(0).toUpperCase() + classification.slice(1);
    } else {
      classification = "Irrelevante"; // fallback padrão
    }
    if (!resume[subject]) {
      resume[subject] = { Boa: 0, Ruim: 0, Irrelevante: 0 }
    }
    if (classification in resume[subject]) {
      resume[subject][classification]++;
    }
  });

  const table = document.createElement("table");
  table.innerHTML = `
      <thead>
        <tr>
          <th>Assunto</th>
          <th>QTD. Boas</th>
          <th>QTD. Ruins</th>
          <th>QTD. Irrelevantes</th>
          <th>Total</th>
        </tr>
      </thead>
      <tbody>
        ${Object.entries(resume).map(([assunto, counts]) => {
    const total = counts.Boa + counts.Ruim + counts.Irrelevante;
    return `
            <tr>
              <td>${assunto}</td>
              <td>${counts.Boa}</td>
              <td>${counts.Ruim}</td>
              <td>${counts.Irrelevante}</td>
              <td>${total}</td>
            </tr>
          `;
  }).join("")}
      </tbody>
    `;
  const graphicDiv = document.getElementById("sectionGraphic");
  const tableDiv = document.getElementById("sectionTable");
  graphicDiv.innerHTML = ""; // limpa conteúdo anterior
  tableDiv.innerHTML = "";
  tableDiv.appendChild(table);


  // Gráfico de barras com Chart.js
  const ctx = document.createElement("canvas");
  graphicDiv.appendChild(ctx);

  const assuntos = Object.keys(resume);
  const totais = assuntos.map(a =>
    resume[a].Boa + resume[a].Ruim + resume[a].Irrelevante
  );
  const boas = assuntos.map(a =>
    resume[a].Boa
  )
  const ruins = assuntos.map(a =>
    resume[a].Ruim
  )
  const irrelevante = assuntos.map(a =>
    resume[a].Irrelevante
  )

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: assuntos,
      datasets: [
        {
          label: 'Noticias Boas',
          data: boas,
          backgroundColor: 'rgba(2, 223, 76, 1)'
        },
        {
          label: 'Noticias Ruins',
          data: ruins,
          backgroundColor: 'rgba(199, 6, 6, 1)'
        },
        {
          label: 'Noticias Irrelevantes',
          data: irrelevante,
          backgroundColor: 'rgb(128, 6, 199)'
        }
      ]

    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          ticks: { stepSize: 1 }
        }
      }
    }
  });
}
function filterInDate() {
  const inicio = document.getElementById("dateStart").value;
  const fim = document.getElementById("dateEnd").value;
  console.log("Filtro aplicado de", inicio, "até", fim);


  const startDate = new Date(inicio + "T00:00:00");
  const endDate = new Date(fim + "T23:59:59");
  if (!startDate || !endDate || isNaN(startDate) || isNaN(endDate)) {
    alert("Por favor, selecione um intervalo de datas válido.");
    return;
  }
  const filteredData = newsJson.filter(n => {
    const dateNews = new Date(n.data);
    return dateNews >= startDate && dateNews <= endDate;
  })

  const filter = document.getElementById("filterDate");
  filter.classList.toggle("visible");
  filter.classList.toggle("hidden");

  const today = new Date().toISOString().split("T")[0];
  document.getElementById("dateStart").value = today;
  document.getElementById("dateEnd").value = today;

  generateTableAndGraphic(filteredData);
}
function filterLast5Days(news) {
  const today = new Date();
  const fiveDaysAgo = new Date();
  today.setHours(0, 0, 0, 0);
  fiveDaysAgo.setHours(0, 0, 0, 0);
  fiveDaysAgo.setDate(today.getDate() - 6);

  const filtered = news.filter(news => {
    const dateNews = new Date(news.data);
    dateNews.setHours(0, 0, 0, 0);
    return dateNews >= fiveDaysAgo && dateNews <= today;
  });
  console.log("Notícias filtradas:", filtered.length);
  return filtered;
}

