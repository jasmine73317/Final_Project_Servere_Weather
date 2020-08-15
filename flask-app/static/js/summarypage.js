d3.json('/resource/', {
    method:"POST",
    body: JSON.stringify({
        //model prediction
      title: 'blah'
    }),
    headers: {
      "Content-type": "application/json; charset=UTF-8"
    }
  })
  .then(json => {
      //send info to the front end
    console.log(json)
  });