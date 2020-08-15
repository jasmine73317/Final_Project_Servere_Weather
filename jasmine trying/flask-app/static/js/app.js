// from data.js
var tableData = data;

// YOUR CODE HERE!
var inputbox = d3.select("input.form-control");
var button = d3.select("buttun#filter-btn");
var form = d3.select("form");
const tablebody = d3.select("tbody");

function runfilter() {
    d3.event.preventDefault();

    let inputtext  = d3.select("input.form-control").property("value");

    console.log(`Do you want to search for ${inputtext}?`);

    let filtertable = tableData.filter(sighting => sighting.datetime.includes(inputtext) );

    console.log(filtertable)

    

    tablebody.html("");
 
    for (var i = 0;  i , filtertable.length; i ++ ) {
        var row = tablebody.append("tr");

        row.append("td").text(filtertable[i].datetime);
        row.append("td").text(filtertable[i].city);
        row.append("td").text(filtertable[i].state);
        row.append("td").text(filtertable[i].country);
        row.append("td").text(filtertable[i].shape);
        row.append("td").text(filtertable[i].durationMinutes);
        row.append("td").text(filtertable[i].comments);
    };
};

inputbox.on("change", runfilter);
button.on("click", runfilter);
form.on("submit", runfilter)

for (var i = 0;  i , tableData.length; i ++ ) {
    var row = tablebody.append("tr");

    row.append("td").text(tableData[i].datetime);
    row.append("td").text(tableData[i].city);
    row.append("td").text(tableData[i].state);
    row.append("td").text(tableData[i].country);
    row.append("td").text(tableData[i].shape);
    row.append("td").text(tableData[i].durationMinutes);
    row.append("td").text(tableData[i].comments);
};